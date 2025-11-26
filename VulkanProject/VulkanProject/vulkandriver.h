//#pragma once
//
//#include "dependencies.h"
//
//class VulkanEngine {
//public:
//	void run();
//
//private: // declare in vulkandriver.cpp
//	void initWindow();
//	void initVulkan();
//	void mainLoop();
//	void cleanup();
//
//	// defaults for window size
//	uint32_t WIDTH = 800;
//	uint32_t HEIGHT = 600;
//
//	// texture and model paths
//	std::vector<std::string> texturePaths;
//	std::vector<std::string> modelPaths;
//
//	// Undeclared variables
//	VkDebugUtilsMessengerEXT debugMessenger;
//	VkDevice device;
//	VkQueue graphicsQueue;
//	VkSurfaceKHR surface;
//	VkQueue presentQueue;
//	VkSwapchainKHR swapChain;
//	std::vector<VkImage> swapChainImages;
//	VkFormat swapChainImageFormat;
//	VkExtent2D swapChainExtent;
//	std::vector<VkImageView> swapChainImageViews;
//	VkDescriptorSetLayout descriptorSetLayout;
//	VkPipelineLayout pipelineLayout;
//	VkRenderPass renderPass;
//	VkPipeline graphicsPipeline;
//	std::vector<VkFramebuffer> swapChainFramebuffers;
//	VkCommandPool commandPool;
//	std::vector<VkCommandBuffer> commandBuffers;
//	std::vector<VkSemaphore> imageAvailableSemaphores;
//	std::vector<VkSemaphore> renderFinishedSemaphores;
//	std::vector<VkFence> inFlightFences;
//	std::vector<Vertex> vertices;
//	std::vector<uint32_t> indices;
//	VkBuffer vertexBuffer;
//	VkDeviceMemory vertexBufferMemory;
//	VkBuffer indexBuffer;
//	VkDeviceMemory indexBufferMemory;
//	VkDescriptorPool descriptorPool;
//	std::vector<VkDescriptorSet> descriptorSets;
//	VkBuffer stagingBuffer;
//	VkDeviceMemory stagingBufferMemory;
//	VkPipelineStageFlags sourceStage;
//	VkPipelineStageFlags destinationStage;
//	VkImage depthImage;
//	VkDeviceMemory depthImageMemory;
//	VkImageView depthImageView;
//	VkImage colorImage;
//	VkDeviceMemory colorImageMemory;
//	VkImageView colorImageView;
//
//	// texture image variables
//	std::vector<uint32_t> mipLevels;
//	std::vector<VkImage> textureImage;
//	std::vector<VkDeviceMemory> textureImageMemory;
//	std::vector<VkImageView> textureImageView;
//	VkSampler textureSampler;
//
//	// Uniform buffer data
//	std::vector<VkBuffer> uniformBuffers;
//	std::vector<VkDeviceMemory> uniformBuffersMemory;
//	std::vector<void*> uniformBuffersMapped;
//
//	// declared variables
//	GLFWwindow* window = nullptr;
//	VkInstance instance = VK_NULL_HANDLE;
//	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
//	const std::vector<const char*> deviceExtensions = {
//			VK_KHR_SWAPCHAIN_EXTENSION_NAME
//	};
//	const int MAX_FRAMES_IN_FLIGHT = 2; // number of frames to be processed concurrently
//	uint32_t currentFrame = 0; // current frame index
//	bool framebufferResized = false; // if we need to update the size of the framebuffer
//	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
//
//	// function declarations
//	void createInstance();
//};