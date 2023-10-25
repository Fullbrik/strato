// SPDX-License-Identifier: MPL-2.0
// Copyright © 2020 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <unistd.h>
#include <dlfcn.h>
#include <unwind.h>
#include <fcntl.h>
#include "signal.h"

namespace skyline::signal {
    thread_local std::exception_ptr SignalExceptionPtr;

    void ExceptionThrow() {
        std::rethrow_exception(SignalExceptionPtr);
    }

    void SleepTillExit() {
        // We don't want to actually exit the process ourselves as it'll automatically be restarted gracefully due to a timeout after being unable to exit within a fixed duration
        while (true)
            sleep(std::numeric_limits<int>::max()); // We just trigger the timeout wait by sleeping forever
    }

    inline StackFrame *SafeFrameRecurse(size_t depth, StackFrame *frame) {
        if (frame) {
            for (size_t it{}; it < depth; it++) {
                if (frame->lr && frame->next)
                    frame = frame->next;
                else
                    SleepTillExit();
            }
        } else {
            SleepTillExit();
        }
        return frame;
    }

    void TerminateHandler() {
        auto exception{std::current_exception()};
        if (exception && exception == SignalExceptionPtr) {
            StackFrame *frame;
            asm("MOV %0, FP" : "=r"(frame));
            frame = SafeFrameRecurse(2, frame); // We unroll past 'std::terminate'

            static void *exceptionThrowEnd{};
            if (!exceptionThrowEnd) {
                // We need to find the function bounds for ExceptionThrow, if we haven't already
                u32 *it{reinterpret_cast<u32 *>(&ExceptionThrow) + 1};
                while (_Unwind_FindEnclosingFunction(it) == &ExceptionThrow)
                    it++;
                exceptionThrowEnd = it - 1;
            }

            auto lookupFrame{frame};
            bool hasAdvanced{};
            while (lookupFrame && lookupFrame->lr) {
                if (lookupFrame->lr >= reinterpret_cast<void *>(&ExceptionThrow) && lookupFrame->lr < exceptionThrowEnd) {
                    // We need to check if the current stack frame is from ExceptionThrow
                    // As we need to skip past it (2 frames) and be able to recognize when we're in an infinite loop
                    if (!hasAdvanced) {
                        frame = SafeFrameRecurse(2, lookupFrame);
                        hasAdvanced = true;
                    } else {
                        SleepTillExit(); // We presumably have no exception handlers left on the stack to consume the exception, it's time to quit
                    }
                }
                lookupFrame = lookupFrame->next;
            }

            if (!frame->next)
                SleepTillExit(); // We don't know the frame's stack boundaries, the only option is to quit

            asm("MOV SP, %x0\n\t" // Stack frame is the first item on a function's stack, it's used to calculate calling function's stack pointer
                "MOV LR, %x1\n\t"
                "MOV FP, %x2\n\t" // The stack frame of the calling function should be set
                "BR %x3"
                : : "r"(frame + 1), "r"(frame->lr), "r"(frame->next), "r"(&ExceptionThrow));

            __builtin_unreachable();
        } else {
            SleepTillExit(); // We don't want to delegate to the older terminate handler as it might cause an exit
        }
    }

    void ExceptionalSignalHandler(int signal, siginfo *info, ucontext *context) {
        SignalException signalException;
        signalException.signal = signal;
        signalException.pc = reinterpret_cast<void *>(context->uc_mcontext.pc);
        if (signal == SIGSEGV)
            signalException.fault = info->si_addr;

        signalException.frames.push_back(reinterpret_cast<void *>(context->uc_mcontext.pc));
        StackFrame *frame{reinterpret_cast<StackFrame *>(context->uc_mcontext.regs[29])};
        while (frame && frame->lr) {
            signalException.frames.push_back(frame->lr);
            frame = frame->next;
        }

        SignalExceptionPtr = std::make_exception_ptr(signalException);
        context->uc_mcontext.pc = reinterpret_cast<u64>(&ExceptionThrow);

        std::set_terminate(TerminateHandler);
    }

    template<typename Signature>
    Signature GetLibcFunction(const char *symbol) {
        void *libc{dlopen("libc.so", RTLD_LOCAL | RTLD_LAZY)};
        if (!libc)
            throw exception("dlopen-ing libc has failed with: {}", dlerror());
        auto function{reinterpret_cast<Signature>(dlsym(libc, symbol))};
        if (!function)
            throw exception("Cannot find '{}' in libc: {}", symbol, dlerror());
        return function;
    }

    void Sigaction(int signal, const struct sigaction *action, struct sigaction *oldAction) {
        static decltype(&sigaction) real{};
        if (!real)
            real = GetLibcFunction<decltype(&sigaction)>("sigaction");
        if (real(signal, action, oldAction) < 0)
            throw exception("sigaction has failed with {}", strerror(errno));
    }

    static void *(*TlsRestorer)(){};

    void SetTlsRestorer(void *(*function)()) {
        TlsRestorer = function;
    }

    struct DefaultSignalHandler {
        void (*function)(int, struct siginfo *, void *){};

        ~DefaultSignalHandler();
    };

    std::array<DefaultSignalHandler, NSIG> DefaultSignalHandlers;

    DefaultSignalHandler::~DefaultSignalHandler() {
        if (function) {
            int signal{static_cast<int>(this - DefaultSignalHandlers.data())};

            struct sigaction oldAction{};
            Sigaction(signal, nullptr, &oldAction);

            struct sigaction action{
                .sa_sigaction = function,
                .sa_flags = oldAction.sa_flags,
            };
            Sigaction(signal, &action);
        }
    }

    thread_local std::array<SignalHandler, NSIG> ThreadGuestSignalHandlers{};
    thread_local std::array<SignalHandler, NSIG> ThreadHostSignalHandlers{};

    /**
     * @brief The first stage handler runs before sigchain, it restores the host TLS and then calls the appropriate handler
     */
    __attribute__((no_stack_protector)) // Stack protector stores data in TLS at the function epilogue and verifies it at the prolog, we cannot allow writes to guest TLS and may switch to an alternative TLS during the signal handler and have disabled the stack protector as a result
    void FirstStageThreadSignalHandler(int signal, siginfo *info, ucontext *context) {
        void *tls{}; // The TLS value prior to being restored if it is
        if (TlsRestorer)
            tls = TlsRestorer();

        auto signum{static_cast<size_t>(signal)};
        auto guestHandler{ThreadGuestSignalHandlers.at(signum)};

        if (guestHandler && tls) // Use the guest handler only if the signal happened in guest code (tls was restored)
            guestHandler(signal, info, context, &tls);
        else if (auto defaultHandler{DefaultSignalHandlers.at(signum).function}) // Use the default handler (sigchain) if present, host handler will be called by it, if any
            defaultHandler(signal, info, context);
        else if (auto hostHandler{ThreadHostSignalHandlers.at(signum)}) // Otherwise use the host handler, if any
            hostHandler(signal, info, context, &tls);
        else [[unlikely]]
            LOGWNF("Unhandled signal {}, PC: 0x{:x}, Fault address: 0x{:x}", signal, context->uc_mcontext.pc, context->uc_mcontext.fault_address);

        if (tls)
            asm volatile("MSR TPIDR_EL0, %x0"::"r"(tls));
    }

    /**
     * @brief The second stage signal handler calls the appropriate handler for the running thread
     */
    __attribute__((no_stack_protector))
    void SecondStageThreadSignalHandler(int signal, siginfo *info, ucontext *context) {
        void *tls{}; // Always nullptr as we don't restore the TLS for signals in host code
        auto hostHandler{ThreadHostSignalHandlers.at(static_cast<size_t>(signal))};
        if (hostHandler)
            hostHandler(signal, info, context, &tls);
    }

    /**
     * @brief Sets up dual-stage signal handling for the given signals, by running the first stage handler before sigchain, or falling back to the default handler which will run the second stage handler
     */
    void SetSignalHandler(std::initializer_list<int> signals, bool syscallRestart) {
        static std::array<std::once_flag, NSIG> signalHandlerOnce{};

        struct sigaction guestAction{
            .sa_sigaction = reinterpret_cast<void (*)(int, siginfo *, void *)>(FirstStageThreadSignalHandler),
            .sa_flags = SA_SIGINFO | SA_EXPOSE_TAGBITS | (syscallRestart ? SA_RESTART : 0) | SA_ONSTACK,
        };

        struct sigaction hostAction{
            .sa_sigaction = reinterpret_cast<void (*)(int, siginfo *, void *)>(SecondStageThreadSignalHandler),
            .sa_flags = SA_SIGINFO | SA_EXPOSE_TAGBITS | (syscallRestart ? SA_RESTART : 0) | SA_ONSTACK,
        };

        for (int signal : signals) {
            std::call_once(signalHandlerOnce[static_cast<size_t>(signal)], [signal, &guestAction, &hostAction]() {
                struct sigaction oldAction{};
                // Set the first stage handler for signals in guest code before sigchain
                Sigaction(signal, &guestAction, &oldAction);
                if (oldAction.sa_flags) {
                    oldAction.sa_flags &= ~SA_UNSUPPORTED; // Mask out kernel not supporting old sigaction() bits
                    oldAction.sa_flags |= SA_SIGINFO | SA_EXPOSE_TAGBITS | SA_RESTART | SA_ONSTACK; // Intentionally ignore these flags for the comparison
                    if (oldAction.sa_flags != (guestAction.sa_flags | SA_RESTART))
                        throw exception("Old sigaction flags aren't equivalent to the replaced signal: {:#b} | {:#b}", oldAction.sa_flags, guestAction.sa_flags);
                }

                DefaultSignalHandlers.at(static_cast<size_t>(signal)).function = (oldAction.sa_flags & SA_SIGINFO) ? oldAction.sa_sigaction : reinterpret_cast<void (*)(int, struct siginfo *, void *)>(oldAction.sa_handler);

                // Set the second stage handler for signals in host code using bionic's sigaction
                // We only need to set the second stage handler if the first stage handler overwrote a default handler
                // Otherwise, the first stage handler will take care of calling the second stage one
                if (DefaultSignalHandlers.at(static_cast<size_t>(signal)).function)
                    sigaction(signal, &hostAction, nullptr);
            });
        }
    }

    void SetGuestSignalHandler(std::initializer_list<int> signals, SignalHandler function, bool syscallRestart) {
        SetSignalHandler(signals, syscallRestart);
        for (int signal : signals)
            ThreadGuestSignalHandlers.at(static_cast<size_t>(signal)) = function;
    }

    void SetHostSignalHandler(std::initializer_list<int> signals, SignalHandler function, bool syscallRestart) {
        SetSignalHandler(signals, syscallRestart);
        for (int signal : signals)
            ThreadHostSignalHandlers.at(static_cast<size_t>(signal)) = function;
    }

    void Sigprocmask(int how, const sigset_t &set, sigset_t *oldSet) {
        static decltype(&pthread_sigmask) real{};
        if (!real)
            real = GetLibcFunction<decltype(&sigprocmask)>("sigprocmask");
        if (real(how, &set, oldSet) < 0)
            throw exception("sigprocmask has failed with {}", strerror(errno));
    }
}
