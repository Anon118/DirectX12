#pragma once
#include "Core/src/win/IWindow.h"
#include "GraphicsError.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include "directx/d3dx12.h"
#include <wrl.h>

namespace chil::app
{
	using Microsoft::WRL::ComPtr;

	int Run(win::IWindow& window)
	{
		// Constants
		constexpr UINT width = 1280;
		constexpr UINT height = 720;
		constexpr UINT bufferCount = 2;
		
		// DXGI factory
		ComPtr<IDXGIFactory4> dxgiFactory;
		CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&dxgiFactory)) >> chk;

		// Device
		ComPtr<ID3D12Device2> device;
		D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)) >> chk;

		// Command queue
		ComPtr<ID3D12CommandQueue> commandQueue;
		{
			const D3D12_COMMAND_QUEUE_DESC desc = {
				.Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
				.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
				.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
				.NodeMask = 0,
			};
			device->CreateCommandQueue(&desc, IID_PPV_ARGS(&commandQueue)) >> chk;
		}

		// Swap chain
		ComPtr<IDXGISwapChain4> swapChain;
		{
			const DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {
				.Width = width,
				.Height = height,
				.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
				.Stereo = FALSE,
				.SampleDesc = {
					.Count = 1,
					.Quality = 0
				},
				.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
				.BufferCount = bufferCount,
				.Scaling = DXGI_SCALING_STRETCH,
				.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
				.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED,
				.Flags = 0 // To set it to DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING, you need to check hardware support.
			};
			ComPtr<IDXGISwapChain1> swapChain1;
			dxgiFactory->CreateSwapChainForHwnd(
				commandQueue.Get(),
				window.GetHandle(),
				&swapChainDesc,
				nullptr,
				nullptr,
				&swapChain1) >> chk;
			swapChain1.As(&swapChain) >> chk;
		}

		// RTV (Render Target View) descriptor heap
		ComPtr<ID3D12DescriptorHeap> rtvDescriptorHeap;
		{
			const D3D12_DESCRIPTOR_HEAP_DESC desc = {
				.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
				.NumDescriptors = bufferCount
			};
			device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&rtvDescriptorHeap)) >> chk;
		}
		const auto rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(
			D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

		// RTV descriptors and buffer references
		ComPtr<ID3D12Resource> backBuffers[bufferCount];
		{
			CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(
				rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());
			for (int i = 0; i < bufferCount; ++i)
			{
				swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])) >> chk;
				device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvHandle);
				rtvHandle.Offset(rtvDescriptorSize);
			}
		}

		// Command allocator
		ComPtr<ID3D12CommandAllocator> commandAllocator;
		device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(&commandAllocator)) >> chk;

		// Command list
		ComPtr<ID3D12GraphicsCommandList> commandList;
		device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
			commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)) >> chk;
		commandList->Close() >> chk;
		
		// Fence
		ComPtr<ID3D12Fence> fence;
		uint64_t fenceValue = 0;
		device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)) >> chk;
		
		// Fence signaling event
		HANDLE fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		if (fenceEvent == NULL) {
			GetLastError() >> chk;
			throw std::runtime_error{ "Failed to create a fence event." };
		}

		// Render loop
		UINT curBackBufferIndex = 0;
		while (!window.IsClosing()) {
			// Advance back buffer
			curBackBufferIndex = swapChain->GetCurrentBackBufferIndex();
			// Select current buffer to render to
			auto& backBuffer = backBuffers[curBackBufferIndex];
			// Reset command list and allocator
			commandAllocator->Reset() >> chk;
			commandList->Reset(commandAllocator.Get(), nullptr) >> chk;
			// Clear the render target
			{
				// Transition buffer resource to render target state
				const auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
					backBuffer.Get(),
					D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
				commandList->ResourceBarrier(1, &barrier);
				// Clear buffer
				FLOAT clearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };
				const CD3DX12_CPU_DESCRIPTOR_HANDLE rtv{
					rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),
					(INT)curBackBufferIndex, rtvDescriptorSize };
				commandList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
			}
			// Prepare buffer for presentation
			{
				const auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
					backBuffer.Get(),
					D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
				commandList->ResourceBarrier(1, &barrier);
			}
			// Submit command list
			{
				commandList->Close() >> chk;
				// Submit command list to command queue
				ID3D12CommandList* const commandLists[] = { commandList.Get() };
				commandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
			}
			// Insert fence to mark command list completion
			commandQueue->Signal(fence.Get(), fenceValue++) >> chk;
			// Present frame
			swapChain->Present(0, 0) >> chk;
			// Wait for command list and command allocator to become free
			fence->SetEventOnCompletion(fenceValue - 1, fenceEvent) >> chk;
			if (::WaitForSingleObject(fenceEvent, INFINITE) == WAIT_FAILED) {
				GetLastError() >> chk;
			}
		}

		return 0;
	}
}