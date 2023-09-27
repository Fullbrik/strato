// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include "command_nodes.h"
#include "gpu/texture/texture.h"
#include <vulkan/vulkan_enums.hpp>

namespace skyline::gpu::interconnect::node {
    RenderPassNode::RenderPassNode(vk::Rect2D renderArea, span<HostTextureView *> pColorAttachments, HostTextureView *pDepthStencilAttachment) : renderArea{renderArea}, colorAttachments{}, depthStencilAttachment{} {
        BindAttachments(pColorAttachments, pDepthStencilAttachment);
    }

    bool RenderPassNode::BindAttachments(span<HostTextureView *> pColorAttachments, HostTextureView *pDepthStencilAttachment) {
        size_t subsetAttachmentCount{std::min(colorAttachments.size(), pColorAttachments.size())};
        bool isColorCompatible{std::equal(colorAttachments.begin(), colorAttachments.begin() + static_cast<ssize_t>(subsetAttachmentCount), pColorAttachments.begin(), pColorAttachments.begin() + static_cast<ssize_t>(subsetAttachmentCount), [](const auto &oldView, const auto &newView) {
            if (oldView && newView && oldView->view != newView)
                return false;
            return true;
        })};
        bool isDepthStencilCompatible{!(depthStencilAttachment && pDepthStencilAttachment && depthStencilAttachment->view != pDepthStencilAttachment)};
        if (!isColorCompatible || !isDepthStencilCompatible)
            // If the attachments aren't a subset of the existing attachments then we can't bind them
            return false;

        if (colorAttachments.size() < pColorAttachments.size()) {
            if (clearValues.size() - 1 == colorAttachments.size()) {
                clearValues.resize(pColorAttachments.size() + 1);
                clearValues[pColorAttachments.size()] = clearValues[colorAttachments.size()];
                clearValues[colorAttachments.size()] = {};
            }

            colorAttachments.resize(pColorAttachments.size());
        }

        for (size_t i{}; i < pColorAttachments.size(); ++i) {
            if (!colorAttachments[i] && pColorAttachments[i])
                colorAttachments[i] = pColorAttachments[i];
        }

        if (!depthStencilAttachment && pDepthStencilAttachment)
            depthStencilAttachment = pDepthStencilAttachment;

        // Note: No need to change the attachments if the new attachments are a subset of the existing attachments

        return true;
    }

    void RenderPassNode::UpdateDependency(vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask) {
        dependencySrcStageMask |= srcStageMask;
        dependencyDstStageMask |= dstStageMask;
    }

    void RenderPassNode::UpdateSelfDependency(vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask, vk::AccessFlags srcAccessMask, vk::AccessFlags dstAccessMask) {
        selfDependencySrcStageMask |= srcStageMask;
        selfDependencyDstStageMask |= dstStageMask;
        selfDependencySrcAccessMask |= srcAccessMask;
        selfDependencyDstAccessMask |= dstAccessMask;
    }

    bool RenderPassNode::ClearColorAttachment(HostTextureView *view, const vk::ClearColorValue &value, GPU &gpu) {
        u32 attachmentIndex{static_cast<u32>(std::distance(colorAttachments.begin(), std::find_if(colorAttachments.begin(), colorAttachments.end(), [&view = view](auto &other){
            return other && view == other->view;
        })))};
        if (colorAttachments.size() == attachmentIndex)
            return false;

        auto &attachment{colorAttachments.at(attachmentIndex)};

        if (attachment && attachment->hasClearValue && clearValues[attachmentIndex].color.uint32 == value.uint32) {
            return true;
        } else if (attachment && attachment->hasClearValue) {
            return false;
        } else {
            if (clearValues.size() < attachmentIndex + 1)
                clearValues.resize(attachmentIndex + 1);
            clearValues[attachmentIndex].color = value;
            attachment->hasClearValue = true;
            return true;
        }
    }

    bool RenderPassNode::ClearDepthStencilAttachment(HostTextureView *view, const vk::ClearDepthStencilValue &value, GPU &gpu) {
        auto &attachment{depthStencilAttachment.value()};
        u32 attachmentIndex{static_cast<u32>(colorAttachments.size())};

        if (attachment.view != view)
            return false;

        if (attachment.hasClearValue && (clearValues[attachmentIndex].depthStencil == value)) {
            return true;
        } else if (attachment.hasClearValue) {
            return false;
        } else {
            if (clearValues.size() < attachmentIndex + 1)
                clearValues.resize(attachmentIndex + 1);
            clearValues[attachmentIndex].depthStencil = value;
            attachment.hasClearValue = true;
            return true;
        }
    }

    vk::RenderPass RenderPassNode::operator()(vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &cycle, GPU &gpu) {
        // TODO: Replace all vector allocations here with a linear allocator
        std::vector<vk::ImageView> vkAttachments;
        std::vector<vk::AttachmentReference> attachmentReferences;
        std::vector<vk::AttachmentDescription> attachmentDescriptions;
        std::vector<vk::FramebufferAttachmentImageInfo> attachmentInfo;
        vk::SubpassDependency selfDependency{};

        size_t attachmentCount{colorAttachments.size() + (depthStencilAttachment ? 1 : 0)};
        vkAttachments.reserve(attachmentCount);
        attachmentReferences.reserve(attachmentCount);
        attachmentDescriptions.reserve(attachmentCount);

        auto addAttachment{[&](const Attachment &attachment) {
            auto &view{attachment.view};
            auto &texture{view->hostTexture};
            vkAttachments.push_back(*view->vkView);
            if (gpu.traits.supportsImagelessFramebuffers)
                attachmentInfo.emplace_back(vk::FramebufferAttachmentImageInfo{
                    .flags = texture->flags,
                    .usage = texture->usage,
                    .width = texture->dimensions.width,
                    .height = texture->dimensions.height,
                    .layerCount = view->range.layerCount,
                    .viewFormatCount = 1,
                    .pViewFormats = &view->format->vkFormat,
                });
            attachmentReferences.emplace_back(vk::AttachmentReference{
                .attachment = static_cast<u32>(attachmentDescriptions.size()),
                .layout = texture->GetLayout(),
            });
            bool hasColorDepth{view->format->vkAspect & (vk::ImageAspectFlagBits::eColor | vk::ImageAspectFlagBits::eDepth)};
            bool hasStencil{view->format->vkAspect & vk::ImageAspectFlagBits::eStencil};
            attachmentDescriptions.emplace_back(vk::AttachmentDescription{
                .format = view->format->vkFormat,
                .samples = texture->sampleCount,
                .loadOp = hasColorDepth ? (attachment.hasClearValue ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad) : vk::AttachmentLoadOp::eDontCare,
                .storeOp = hasColorDepth ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare,
                .stencilLoadOp = hasStencil ? (attachment.hasClearValue ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad) : vk::AttachmentLoadOp::eDontCare,
                .stencilStoreOp = hasStencil ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare,
                .initialLayout = texture->GetLayout(),
                .finalLayout = texture->GetLayout(),
            });
        }};

        for (const auto &attachment : colorAttachments) {
            if (attachment && attachment->view && attachment->view->texture)
                addAttachment(*attachment);
            else if (attachment && attachment->view && !attachment->view->texture) {
                Logger::Error("Destroyed texture at render pass time!");
                attachmentReferences.emplace_back(vk::AttachmentReference{
                    .attachment = VK_ATTACHMENT_UNUSED,
                    .layout = vk::ImageLayout::eUndefined,
                });
            } else
                attachmentReferences.emplace_back(vk::AttachmentReference{
                    .attachment = VK_ATTACHMENT_UNUSED,
                    .layout = vk::ImageLayout::eUndefined,
                });
        }

        if (depthStencilAttachment)
            addAttachment(*depthStencilAttachment);

        u32 colorAttachmentCount{static_cast<u32>(colorAttachments.size())};
        vk::SubpassDescription subpassDescription{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = colorAttachmentCount,
            .pColorAttachments = reinterpret_cast<vk::AttachmentReference *>(attachmentReferences.data()),
            .pDepthStencilAttachment = reinterpret_cast<vk::AttachmentReference *>(depthStencilAttachment ? attachmentReferences.data() + colorAttachmentCount : nullptr),
        };

        if (dependencyDstStageMask && dependencySrcStageMask) {
            commandBuffer.pipelineBarrier(dependencySrcStageMask, dependencyDstStageMask, {}, vk::MemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
            }, {}, {});
        }

        if (selfDependencySrcStageMask && selfDependencyDstStageMask) {
            selfDependency.srcStageMask = selfDependencySrcStageMask;
            selfDependency.dstStageMask = selfDependencyDstStageMask;
            selfDependency.srcAccessMask = selfDependencySrcAccessMask;
            selfDependency.dstAccessMask = selfDependencyDstAccessMask;
            selfDependency.dependencyFlags = vk::DependencyFlagBits::eByRegion;
        }

        auto renderPass{gpu.renderPassCache.GetRenderPass(vk::RenderPassCreateInfo{
            .attachmentCount = static_cast<u32>(attachmentDescriptions.size()),
            .pAttachments = attachmentDescriptions.data(),
            .subpassCount = 1,
            .pSubpasses = &subpassDescription,
            .dependencyCount = selfDependency.dependencyFlags ? 1U : 0U,
            .pDependencies = selfDependency.dependencyFlags ? &selfDependency : nullptr
        })};

        bool useImagelessFramebuffer{gpu.traits.supportsImagelessFramebuffers};
        cache::FramebufferCreateInfo framebufferCreateInfo{
            vk::FramebufferCreateInfo{
                .flags = useImagelessFramebuffer ? vk::FramebufferCreateFlagBits::eImageless : vk::FramebufferCreateFlags{},
                .renderPass = renderPass,
                .attachmentCount = static_cast<u32>(vkAttachments.size()),
                .pAttachments = vkAttachments.data(),
                .width = renderArea.extent.width + static_cast<u32>(renderArea.offset.x),
                .height = renderArea.extent.height + static_cast<u32>(renderArea.offset.y),
                .layers = 1,
            },
            vk::FramebufferAttachmentsCreateInfo{
                .attachmentImageInfoCount = static_cast<u32>(attachmentInfo.size()),
                .pAttachmentImageInfos = attachmentInfo.data(),
            }
        };

        if (!useImagelessFramebuffer)
            framebufferCreateInfo.unlink<vk::FramebufferAttachmentsCreateInfo>();

        auto framebuffer{gpu.framebufferCache.GetFramebuffer(framebufferCreateInfo)};

        vk::StructureChain<vk::RenderPassBeginInfo, vk::RenderPassAttachmentBeginInfo> renderPassBeginInfo{
            vk::RenderPassBeginInfo{
                .renderPass = renderPass,
                .framebuffer = framebuffer,
                .renderArea = renderArea,
                .clearValueCount = static_cast<u32>(clearValues.size()),
                .pClearValues = clearValues.data(),
            },
            vk::RenderPassAttachmentBeginInfo{
                .attachmentCount = static_cast<u32>(vkAttachments.size()),
                .pAttachments = vkAttachments.data(),
            }
        };

        if (!useImagelessFramebuffer)
            renderPassBeginInfo.unlink<vk::RenderPassAttachmentBeginInfo>();

        commandBuffer.beginRenderPass(renderPassBeginInfo.get<vk::RenderPassBeginInfo>(), vk::SubpassContents::eInline);

        return renderPass;
    }

    void SyncNode::operator()(vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &cycle, GPU &gpu) {
        if (active)
            commandBuffer.pipelineBarrier(srcStages, dstStages, deps, {}, vk::ArrayProxy<vk::BufferMemoryBarrier>(static_cast<u32>(bufferBarriers.size()), bufferBarriers.data()), vk::ArrayProxy<vk::ImageMemoryBarrier>(static_cast<u32>(imageBarriers.size()), imageBarriers.data()));
    }
}
