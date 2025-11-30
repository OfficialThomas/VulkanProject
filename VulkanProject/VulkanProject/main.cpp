#define NOMINMAX // to prevent windows.h from defining min and max macros
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION // texture loading library implementation
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <map>
#include <optional>
#include <set>
#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <fstream>
#include <array>
#include <unordered_map>
#include <atomic>
#include <iomanip>  // For std::setprecision

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::string MODEL_PATH = "models/XWing_Woody.obj";
const std::string TEXTURE_DIR = "textures/";  // Keep this for fallback, but we'll also check "Textures/"

const std::vector<std::string> TEXTURE_PATHS = {
	"textures/Engines_WingsColor.tga",
	"textures/Fuselage_CockpitColor.tga"
};

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;
	uint32_t textureId;  // ADD THIS LINE

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		// Position
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		// Normal
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, normal);

		// TexCoord
		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		// TextureId - ADD THIS
		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32_UINT;
		attributeDescriptions[3].offset = offsetof(Vertex, textureId);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos &&
			normal == other.normal &&
			texCoord == other.texCoord &&
			textureId == other.textureId;
	}
};

// hash function - for the vertex
namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1) ^
				(hash<uint32_t>()(vertex.textureId) << 1);  // new for multiple textures
		}
	};
}

struct Sphere {
	glm::vec3 pos;
	glm::vec3 color;
	float radius;
};

struct UniformBufferObject {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	// Debug tracking
	std::atomic<int> activeCommandBuffers{0};
	static constexpr int MAX_SINGLE_TIME_COMMAND_BUFFERS = 16;  // Reasonable limit

	// Undeclared variables
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkDevice device;
	VkQueue graphicsQueue;
	VkSurfaceKHR surface;
	VkQueue presentQueue;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	VkPipelineLayout pipelineLayout;
	VkRenderPass renderPass;
	VkPipeline graphicsPipeline;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;
	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;

	// THIS WILL ALL BE CHANGED TO VECTOR ARRAYS
	// 1st texture
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	uint32_t mipLevels;

	// 2nd texture
	VkImage textureImage2;
	VkDeviceMemory textureImageMemory2;
	VkImageView textureImageView2;
	uint32_t mipLevels2;

	// New array vectors for images - delete values above once this works
	std::vector<VkImage> textureImages;
	std::vector<VkImageView> textureImageViews;
	std::vector<VkDeviceMemory> textureImageMemories;
	VkSampler textureSampler;  // Can share one sampler for all textures

	std::vector<VkDescriptorSet> textureDescriptorSets;  // One descriptor set per texture

	struct DrawCommand { // how and where to draw textures
		uint32_t indexCount;
		uint32_t firstIndex;
		uint32_t textureIndex;
	};

	std::vector<DrawCommand> drawCommands;

	// Uniform buffer data
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	// Command buffer variables
	std::vector<VkCommandBuffer> commandBuffers;
	uint32_t currentFrame = 0;
	const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

	// DescriptorSetLayout variables (separated camera from textures)
	VkDescriptorSetLayout descriptorSetLayout;        // Set 0: Camera UBO (existing)
	VkDescriptorSetLayout textureDescriptorSetLayout; // Set 1: Texture samplers (NEW)

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	const std::vector<const char*> deviceExtensions = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};
	bool framebufferResized = false; // if we need to update the size of the framebuffer
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	void initVulkan() {
		// Debug file structure first
		debugFileStructure();
		
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();        // Camera layout
		createTextureDescriptorSetLayout(); // NEW - Texture layout
		createGraphicsPipeline();           // Now uses both layouts
		createCommandPool();                // ENSURE this is before anything that needs command buffers
		createColorResources();             // Uses command buffers
		createDepthResources();             // Uses command buffers  
		createFramebuffers();
		createTextureSampler();
		createDefaultTexture();             // Uses command buffers - moved after command pool
		loadModel();                        // Uses command buffers - moved after command pool
		createVertexBuffer();               
		createIndexBuffer();                 	
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createTextureDescriptorSets();
		organizeDrawCommands();
		createCommandBuffers();
		createSyncObjects();
	}

	void organizeDrawCommands() {
		drawCommands.clear();

		if (indices.empty()) {
			std::cout << "ERROR: No indices to organize!" << std::endl;
			return;
		}

		std::cout << "\n=== DRAW COMMAND ORGANIZATION ===" << std::endl;
		std::cout << "Organizing " << indices.size() << " indices into draw commands..." << std::endl;
		std::cout << "Total vertices: " << vertices.size() << std::endl;
		std::cout << "Available textures: " << textureImages.size() << std::endl;

		// Analyze texture distribution first
		std::map<uint32_t, uint32_t> textureUsageCount;
		for (size_t i = 0; i < indices.size(); i++) {
			if (indices[i] < vertices.size()) {
				uint32_t textureId = vertices[indices[i]].textureId;
				textureUsageCount[textureId]++;
			}
		}

		std::cout << "\nTexture usage distribution:" << std::endl;
		for (const auto& pair : textureUsageCount) {
			std::cout << "  Texture " << pair.first << ": " << pair.second << " index references" << std::endl;
		}

		// Group consecutive triangles by texture
		uint32_t currentTexture = vertices[indices[0]].textureId;
		uint32_t startIndex = 0;
		uint32_t totalIndicesProcessed = 0;

		std::cout << "\nGenerating draw commands:" << std::endl;
		std::cout << "Starting with texture " << currentTexture << std::endl;

		for (uint32_t i = 3; i <= indices.size(); i += 3) {  // Step by triangles (3 indices)
			bool newGroup = false;

			if (i >= indices.size()) {
				newGroup = true;
			}
			else if (vertices[indices[i]].textureId != currentTexture) {
				newGroup = true;
			}

			if (newGroup) {
				DrawCommand cmd;
				cmd.firstIndex = startIndex;
				cmd.indexCount = i - startIndex;
				cmd.textureIndex = currentTexture;
				drawCommands.push_back(cmd);

				std::cout << "Draw command " << (drawCommands.size() - 1) 
						  << ": indices " << startIndex << "-" << (startIndex + cmd.indexCount - 1) 
						  << " (count: " << cmd.indexCount << ", triangles: " << cmd.indexCount / 3 
						  << ") using texture " << cmd.textureIndex << std::endl;

				totalIndicesProcessed += cmd.indexCount;

				if (i < indices.size()) {
					currentTexture = vertices[indices[i]].textureId;
					startIndex = i;
					std::cout << "  Next group starts at index " << i << " with texture " << currentTexture << std::endl;
				}
			}
		}

		std::cout << "\nDraw command summary:" << std::endl;
		std::cout << "Created " << drawCommands.size() << " draw commands" << std::endl;
		std::cout << "Total indices in commands: " << totalIndicesProcessed << " / " << indices.size() << std::endl;
		
		// Validate draw commands
		bool hasErrors = false;
		for (size_t i = 0; i < drawCommands.size(); i++) {
			const auto& cmd = drawCommands[i];
			
			// Check texture index bounds
			if (cmd.textureIndex >= textureImages.size()) {
				std::cout << "ERROR: Draw command " << i << " references texture " << cmd.textureIndex 
						  << " but only have " << textureImages.size() << " textures!" << std::endl;
				hasErrors = true;
			}
			
			// Check index bounds
			if (cmd.firstIndex + cmd.indexCount > indices.size()) {
				std::cout << "ERROR: Draw command " << i << " accesses indices beyond bounds: " 
						  << cmd.firstIndex << "+" << cmd.indexCount << " > " << indices.size() << std::endl;
				hasErrors = true;
			}
			
			// Check if triangle count is valid
			if (cmd.indexCount % 3 != 0) {
				std::cout << "WARNING: Draw command " << i << " has " << cmd.indexCount 
						  << " indices (not divisible by 3)" << std::endl;
			}
		}
		
		if (totalIndicesProcessed != indices.size()) {
			std::cout << "ERROR: Draw commands cover " << totalIndicesProcessed 
					  << " indices but total is " << indices.size() << " (missing " 
					  << (indices.size() - totalIndicesProcessed) << " indices)" << std::endl;
			hasErrors = true;
		}

		if (hasErrors) {
			std::cout << "CRITICAL: Draw command organization has errors - some faces may not render!" << std::endl;
		} else {
			std::cout << "SUCCESS: Draw command organization validated successfully" << std::endl;
		}
		
		std::cout << "==================================\n" << std::endl;
	}

	void createColorResources() {
		VkFormat colorFormat = swapChainImageFormat;

		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
		colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	// Function to get the maximum usable sample count for MSAA
	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
		if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
		if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}

	// Mipmaps generation function
	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
		// Check if image format supports linear blitting
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);
		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			// transition mip (i-1) from TRANSFER_DST -> TRANSFER_SRC
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(commandBuffer,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR);

			// transition mip (i-1) from TRANSFER_SRC -> SHADER_READ_ONLY
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		// transition last mip level to SHADER_READ_ONLY
		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	void createDefaultTexture() {
		// Create a simple 1x1 white texture as texture 0
		uint8_t whitePixel[4] = { 255, 255, 255, 255 }; // RGBA white

		VkDeviceSize imageSize = 4; // 4 bytes for 1 pixel

		// Create staging buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, whitePixel, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		// Make sure vectors are large enough
		textureImages.resize(1);
		textureImageMemories.resize(1);
		textureImageViews.resize(1);

		// Create image - use VK_SAMPLE_COUNT_1_BIT for textures (not msaaSamples)
		createImage(1, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			textureImages[0], textureImageMemories[0]);

		// Transition and copy
		transitionImageLayout(textureImages[0], VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImages[0], 1, 1);
		transitionImageLayout(textureImages[0], VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		// Create image view
		textureImageViews[0] = createImageView(textureImages[0],
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_ASPECT_COLOR_BIT);

		std::cout << "Created default white texture at index 0" << std::endl;
	}

	void createColoredTexture(float r, float g, float b, uint32_t textureIndex) {
		// Create a 1x1 colored texture
		uint8_t colorPixel[4] = { 
			static_cast<uint8_t>(r * 255), 
			static_cast<uint8_t>(g * 255), 
			static_cast<uint8_t>(b * 255), 
			255 
		}; // RGBA

		VkDeviceSize imageSize = 4; // 4 bytes for 1 pixel

		// Create staging buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, colorPixel, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		// Make sure vectors are large enough
		if (textureIndex >= textureImages.size()) {
			textureImages.resize(textureIndex + 1);
			textureImageMemories.resize(textureIndex + 1);
			textureImageViews.resize(textureIndex + 1);
		}

		// Create image
		createImage(1, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			textureImages[textureIndex], textureImageMemories[textureIndex]);

		// Transition and copy
		transitionImageLayout(textureImages[textureIndex], VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImages[textureIndex], 1, 1);
		transitionImageLayout(textureImages[textureIndex], VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		// Create image view
		textureImageViews[textureIndex] = createImageView(textureImages[textureIndex],
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_ASPECT_COLOR_BIT);

		std::cout << "Created colored texture at index " << textureIndex 
			<< " with color RGB(" << static_cast<int>(colorPixel[0]) << "," 
			<< static_cast<int>(colorPixel[1]) << "," 
			<< static_cast<int>(colorPixel[2]) << ")" << std::endl;
	}

	void loadModel() {
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		// Extract the directory from the model path for relative MTL loading
		std::string modelDir = "";
		size_t lastSlash = MODEL_PATH.find_last_of('/');
		if (lastSlash != std::string::npos) {
			modelDir = MODEL_PATH.substr(0, lastSlash + 1);
		}
		
		std::cout << "Loading model from: " << MODEL_PATH << std::endl;
		std::cout << "Model directory: " << modelDir << std::endl;

		// Load with material directory specified
		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str(), modelDir.c_str())) {
			throw std::runtime_error(warn + err);
		}
		
		if (!warn.empty()) {
			std::cout << "Warning: " << warn << std::endl;
		}
		if (!err.empty()) {
			std::cout << "Error: " << err << std::endl;
		}

		// Create colored textures based on material properties when texture files are missing
		std::map<std::string, uint32_t> textureMap;
		uint32_t nextTextureId = 1;  // Start at 1 instead of 0 (0 is default white)

		std::cout << "Found " << materials.size() << " materials:" << std::endl;
		for (size_t i = 0; i < materials.size(); i++) {
			const auto& material = materials[i];
			std::cout << "  Material " << i << ": " << material.name << std::endl;
			std::cout << "    Color: R=" << material.diffuse[0] << " G=" << material.diffuse[1] << " B=" << material.diffuse[2] << std::endl;
			
			if (!material.diffuse_texname.empty()) {
				std::cout << "    Original texture path: " << material.diffuse_texname << std::endl;
				
				// Check if we've already processed this texture
				if (textureMap.find(material.diffuse_texname) == textureMap.end()) {
					// Find the correct texture path
					std::string texturePath = findTexturePath(material.diffuse_texname);
					std::cout << "    Resolved texture path: " << texturePath << std::endl;

					try {
						createTextureImage(texturePath, nextTextureId);
						textureMap[material.diffuse_texname] = nextTextureId;
						nextTextureId++;
						std::cout << "    Successfully loaded texture at index " << (nextTextureId-1) << std::endl;
					}
					catch (const std::exception& e) {
						std::cout << "    Failed to load texture: " << e.what() << std::endl;
						
						// Create a colored texture based on material diffuse color
						std::cout << "    Creating colored texture based on material color" << std::endl;
						try {
							createColoredTexture(material.diffuse[0], material.diffuse[1], material.diffuse[2], nextTextureId);
							textureMap[material.diffuse_texname] = nextTextureId;
							nextTextureId++;
							std::cout << "    Successfully created colored texture at index " << (nextTextureId-1) << std::endl;
						}
						catch (const std::exception& e2) {
							std::cout << "    Failed to create colored texture: " << e2.what() << std::endl;
							std::cout << "    Using default white texture" << std::endl;
							textureMap[material.diffuse_texname] = 0;  // Use default white texture
						}
					}
				}
			} else {
				std::cout << "    No texture specified - creating colored texture based on material" << std::endl;
				// Create a unique texture for this material based on its color
				std::string materialKey = "material_" + std::to_string(i);
				if (textureMap.find(materialKey) == textureMap.end()) {
					try {
						createColoredTexture(material.diffuse[0], material.diffuse[1], material.diffuse[2], nextTextureId);
						textureMap[materialKey] = nextTextureId;
						nextTextureId++;
						std::cout << "    Created colored texture at index " << (nextTextureId-1) << std::endl;
					}
					catch (const std::exception& e) {
						std::cout << "    Failed to create colored texture: " << e.what() << std::endl;
						textureMap[materialKey] = 0;  // Use default white texture
					}
				}
			}
		}

		std::cout << "Created " << (nextTextureId-1) << " textures total" << std::endl;

		// Count total faces first
		size_t totalFaces = 0;
		for (const auto& shape : shapes) {
			totalFaces += shape.mesh.num_face_vertices.size();
		}
		
		std::cout << "Processing " << totalFaces << " faces..." << std::endl;
		
		// Safety check for very large models
		if (totalFaces > 100000) {
			std::cout << "WARNING: Large model detected (" << totalFaces << " faces). This may take a while..." << std::endl;
		}

		// Now load vertices with texture IDs
		std::unordered_map<Vertex, uint32_t> uniqueVertices{};
		
		size_t processedFaces = 0;
		size_t progressInterval = std::max(totalFaces / 100, (size_t)1000); // Print progress every 1% or 1000 faces

		for (const auto& shape : shapes) {
			size_t index_offset = 0;

			for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
				int fv = shape.mesh.num_face_vertices[f];

				// Get material for this face - add bounds checking
				int materialId = -1;
				if (f < shape.mesh.material_ids.size()) {
					materialId = shape.mesh.material_ids[f];
				}
				uint32_t textureId = 0;  // Default texture

				if (materialId >= 0 && materialId < (int)materials.size()) {
					const auto& material = materials[materialId];
					
					// Only print detailed debug info for first few faces
					if (processedFaces < 10) {
						std::cout << "Face " << processedFaces << " uses material " << materialId << " (" << material.name << ")" << std::endl;
					}
					
					if (!material.diffuse_texname.empty() && textureMap.count(material.diffuse_texname)) {
						textureId = textureMap[material.diffuse_texname];
						if (processedFaces < 10) {
							std::cout << "  Using texture " << textureId << " for material texture: " << material.diffuse_texname << std::endl;
						}
					} else {
						// Use material-based texture
						std::string materialKey = "material_" + std::to_string(materialId);
						if (textureMap.count(materialKey)) {
							textureId = textureMap[materialKey];
							if (processedFaces < 10) {
								std::cout << "  Using texture " << textureId << " for material color" << std::endl;
							}
						}
					}
				} else {
					if (processedFaces < 10) {
						std::cout << "Face " << processedFaces << " has invalid/missing material ID: " << materialId << std::endl;
					}
				}

				// Handle both triangles and quads by triangulating
				if (fv == 3) {
					// Triangle - process normally
					for (size_t v = 0; v < 3; v++) {
						tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

						Vertex vertex{};

						vertex.pos = {
							attrib.vertices[3 * idx.vertex_index + 0],
							attrib.vertices[3 * idx.vertex_index + 1],
							attrib.vertices[3 * idx.vertex_index + 2]
						};

						if (idx.normal_index >= 0) {
							vertex.normal = {
								attrib.normals[3 * idx.normal_index + 0],
								attrib.normals[3 * idx.normal_index + 1],
								attrib.normals[3 * idx.normal_index + 2]
							};
						} else {
							// Calculate normal if not provided - this might be missing!
							vertex.normal = {0.0f, 0.0f, 1.0f};  // Default upward normal
						}

						if (idx.texcoord_index >= 0) {
							vertex.texCoord = {
								attrib.texcoords[2 * idx.texcoord_index + 0],
								1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
							};
						}

						vertex.textureId = textureId;

						if (uniqueVertices.count(vertex) == 0) {
							uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
							vertices.push_back(vertex);
						}

						indices.push_back(uniqueVertices[vertex]);
					}
				} else if (fv == 4) {
					// Quad - triangulate into two triangles
					// First triangle: 0, 1, 2
					// Second triangle: 0, 2, 3
					std::vector<size_t> quadIndices = {0, 1, 2, 0, 2, 3};
					
					for (size_t triIdx : quadIndices) {
						tinyobj::index_t idx = shape.mesh.indices[index_offset + triIdx];

						Vertex vertex{};

						vertex.pos = {
							attrib.vertices[3 * idx.vertex_index + 0],
							attrib.vertices[3 * idx.vertex_index + 1],
							attrib.vertices[3 * idx.vertex_index + 2]
						};

						if (idx.normal_index >= 0) {
							vertex.normal = {
								attrib.normals[3 * idx.normal_index + 0],
								attrib.normals[3 * idx.normal_index + 1],
								attrib.normals[3 * idx.normal_index + 2]
							};
						} else {
							// Default normal for missing normals
							vertex.normal = {0.0f, 0.0f, 1.0f};
						}

						if (idx.texcoord_index >= 0) {
							vertex.texCoord = {
								attrib.texcoords[2 * idx.texcoord_index + 0],
								1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
							};
						}

						vertex.textureId = textureId;

						if (uniqueVertices.count(vertex) == 0) {
							uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
							vertices.push_back(vertex);
						}

						indices.push_back(uniqueVertices[vertex]);
					}
				} else if (fv > 4) {
					// Polygon with more than 4 vertices - triangulate using fan triangulation
					// Create triangles: (0,1,2), (0,2,3), (0,3,4), etc.
					for (size_t v = 1; v < fv - 1; v++) {
						std::vector<size_t> triIndices = {0, v, v + 1};
						
						for (size_t triIdx : triIndices) {
							tinyobj::index_t idx = shape.mesh.indices[index_offset + triIdx];

							Vertex vertex{};

							vertex.pos = {
								attrib.vertices[3 * idx.vertex_index + 0],
								attrib.vertices[3 * idx.vertex_index + 1],
								attrib.vertices[3 * idx.vertex_index + 2]
							};

							if (idx.normal_index >= 0) {
								vertex.normal = {
									attrib.normals[3 * idx.normal_index + 0],
									attrib.normals[3 * idx.normal_index + 1],
									attrib.normals[3 * idx.normal_index + 2]
								};
							} else {
								// Default normal for missing normals  
								vertex.normal = {0.0f, 0.0f, 1.0f};
							}

							if (idx.texcoord_index >= 0) {
								vertex.texCoord = {
									attrib.texcoords[2 * idx.texcoord_index + 0],
									1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]
								};
							}

							vertex.textureId = textureId;

							if (uniqueVertices.count(vertex) == 0) {
								uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
								vertices.push_back(vertex);
							}

							indices.push_back(uniqueVertices[vertex]);
						}
					}
				}

				index_offset += fv;
				processedFaces++;
				
				// Print progress periodically
				if (processedFaces % progressInterval == 0) {
					float progress = (float)processedFaces / totalFaces * 100.0f;
					std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "% (" << processedFaces << "/" << totalFaces << " faces)" << std::endl;
				}
			}
		}

		std::cout << "Loaded " << vertices.size() << " unique vertices and " << indices.size() << " indices" << std::endl;
		
		// Debug: Check for vertices with missing normals and texture coordinates
		uint32_t verticesWithDefaultNormals = 0;
		uint32_t verticesWithDefaultTexCoords = 0;
		uint32_t verticesWithInvalidTextureIds = 0;
		std::map<uint32_t, uint32_t> textureIdCounts;
		
		for (const auto& vertex : vertices) {
			// Check for default normals (what we assign when normals are missing)
			if (vertex.normal.x == 0.0f && vertex.normal.y == 0.0f && vertex.normal.z == 1.0f) {
				verticesWithDefaultNormals++;
			}
			
			// Check for default texture coordinates
			if (vertex.texCoord.x == 0.0f && vertex.texCoord.y == 0.0f) {
				verticesWithDefaultTexCoords++;
			}
			
			// Check for invalid texture IDs
			if (vertex.textureId >= textureImages.size()) {
				verticesWithInvalidTextureIds++;
			}
			
			textureIdCounts[vertex.textureId]++;
		}
		
		std::cout << "\nVertex validation results:" << std::endl;
		std::cout << "  Vertices with default normals: " << verticesWithDefaultNormals << " / " << vertices.size() 
		          << " (" << (float)verticesWithDefaultNormals / vertices.size() * 100.0f << "%)" << std::endl;
		std::cout << "  Vertices with default tex coords: " << verticesWithDefaultTexCoords << " / " << vertices.size() 
		          << " (" << (float)verticesWithDefaultTexCoords / vertices.size() * 100.0f << "%)" << std::endl;
		std::cout << "  Vertices with invalid texture IDs: " << verticesWithInvalidTextureIds << " / " << vertices.size() << std::endl;
		
		if (verticesWithInvalidTextureIds > 0) {
			std::cout << "  WARNING: Some vertices reference non-existent textures!" << std::endl;
		}
		
		std::cout << "\nTexture ID distribution in vertices:" << std::endl;
		for (const auto& pair : textureIdCounts) {
			std::cout << "    Texture " << pair.first << ": " << pair.second << " vertices";
			if (pair.first >= textureImages.size()) {
				std::cout << " (INVALID - texture doesn't exist!)";
			}
			std::cout << std::endl;
		}
		
		// Safety check for memory usage
		size_t estimatedMemoryMB = (vertices.size() * sizeof(Vertex) + indices.size() * sizeof(uint32_t)) / (1024 * 1024);
		std::cout << "\nMemory usage: " << estimatedMemoryMB << " MB" << std::endl;
		
		if (estimatedMemoryMB > 500) {
			std::cout << "WARNING: High memory usage detected. Consider using a simpler model for testing." << std::endl;
		}
		
		// Debug: Check for degenerate triangles
		uint32_t degenerateTriangles = 0;
		for (size_t i = 0; i < indices.size(); i += 3) {
			if (i + 2 < indices.size()) {
				uint32_t v1 = indices[i];
				uint32_t v2 = indices[i + 1]; 
				uint32_t v3 = indices[i + 2];
				
				if (v1 == v2 || v1 == v3 || v2 == v3) {
					degenerateTriangles++;
				}
			}
		}

		if (degenerateTriangles > 0) {
			std::cout << "WARNING: Found " << degenerateTriangles << " degenerate triangles!" << std::endl;
		}
	}

	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

		transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	void createTextureSampler() {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create texture sampler!");
		}
	}

	void createTextureImageViews() {
		// Resize the vector to hold all image views
		textureImageViews.resize(textureImages.size());

		// Create an image view for each texture
		for (size_t i = 0; i < textureImages.size(); i++) {
			textureImageViews[i] = createImageView(textureImages[i],
				VK_FORMAT_R8G8B8A8_SRGB,
				VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		//viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image view!");
		}

		return imageView;
	}

	void createTextureImage(const std::string& texturePath, uint32_t textureIndex) {
		std::cout << "Attempting to load texture from: " << texturePath << std::endl;
		
		// Check if file exists first
		if (!fileExists(texturePath)) {
			std::string error = "Texture file not found: " + texturePath;
			std::cout << "ERROR: " << error << std::endl;
			throw std::runtime_error(error);
		}
		
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		
		if (!pixels) {
			std::string error = "Failed to load texture image: " + texturePath + " - " + std::string(stbi_failure_reason());
			std::cout << "ERROR: " << error << std::endl;
			throw std::runtime_error(error);
		}

		std::cout << "Successfully loaded texture: " << texturePath << " (" << texWidth << "x" << texHeight << ", " << texChannels << " channels)" << std::endl;

		VkDeviceSize imageSize = texWidth * texHeight * 4;

		// Create staging buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);

		// Make sure vectors are large enough - resize
		if (textureIndex >= textureImages.size()) {
			textureImages.resize(textureIndex + 1);
			textureImageMemories.resize(textureIndex + 1);
			textureImageViews.resize(textureIndex + 1);
		}

		// Create image - use VK_SAMPLE_COUNT_1_BIT for textures (not msaaSamples)
		createImage(texWidth, texHeight,
			1,                              // mipLevels - 1 for now (can add mipmapping later)
			VK_SAMPLE_COUNT_1_BIT,          // Textures should use VK_SAMPLE_COUNT_1_BIT, not msaaSamples
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			textureImages[textureIndex],
			textureImageMemories[textureIndex]);

		// Transition and copy
		transitionImageLayout(textureImages[textureIndex], VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImages[textureIndex],
			static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImages[textureIndex], VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		// Create image view
		textureImageViews[textureIndex] = createImageView(textureImages[textureIndex],
			VK_FORMAT_R8G8B8A8_SRGB,
			VK_IMAGE_ASPECT_COLOR_BIT);

		std::cout << "Successfully created texture at index " << textureIndex << std::endl;
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
		try {
			commandBuffer = beginSingleTimeCommands();

			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;

			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;

			region.imageOffset = { 0, 0, 0 };
			region.imageExtent = {
				width,
				height,
				1
			};

			vkCmdCopyBufferToImage(
				commandBuffer,
				buffer,
				image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1,
				&region
			);

			endSingleTimeCommands(commandBuffer);
		}
		catch (const std::exception& e) {
			std::cerr << "Error in copyBufferToImage: " << e.what() << std::endl;
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
			activeCommandBuffers.fetch_sub(1);
			throw;
		}
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
		VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
		try {
			commandBuffer = beginSingleTimeCommands();

			VkImageMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barrier.oldLayout = oldLayout;
			barrier.newLayout = newLayout;

			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			barrier.image = image;

			// set aspect mask based on format (depth/stencil vs color)
			if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT ||
				format == VK_FORMAT_D32_SFLOAT) {
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
				// if stencil present, include it:
				if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
					barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
				}
			}
			else {
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			}

			if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

				if (hasStencilComponent(format)) {
					barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
				}
			}
			else {
				barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			}
			
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;

			VkPipelineStageFlags sourceStage;
			VkPipelineStageFlags destinationStage;

			if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			}
			else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

				sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
				destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			}
			else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
				barrier.srcAccessMask = 0;
				barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

				sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
				destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			}
			else {
				throw std::invalid_argument("unsupported layout transition!");
			}

			vkCmdPipelineBarrier(
				commandBuffer,
				sourceStage, destinationStage,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);

			endSingleTimeCommands(commandBuffer);
		}
		catch (const std::exception& e) {
			std::cerr << "Error in transitionImageLayout: " << e.what() << std::endl;
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
			activeCommandBuffers.fetch_sub(1);
			throw;
		}
	}

	VkCommandBuffer beginSingleTimeCommands() {
		if (device == VK_NULL_HANDLE) {
			std::cerr << "Device is VK_NULL_HANDLE in beginSingleTimeCommands!" << std::endl;
			throw std::runtime_error("Device is null");
		}
		if (commandPool == VK_NULL_HANDLE) {
			std::cerr << "Command pool is VK_NULL_HANDLE in beginSingleTimeCommands!" << std::endl;
			throw std::runtime_error("Command pool is null");
		}

		int currentActive = activeCommandBuffers.load();
		if (currentActive >= MAX_SINGLE_TIME_COMMAND_BUFFERS) {
			std::cerr << "Too many active single-time command buffers (" << currentActive 
					  << " >= " << MAX_SINGLE_TIME_COMMAND_BUFFERS << "). Possible memory leak!" << std::endl;
			throw std::runtime_error("Command buffer pool exhausted - possible memory leak");
		}

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		// Only print debug info if there are issues or if we have very few active buffers
		if (currentActive == 0) {
			std::cout << "Allocating first single-time command buffer from pool" << std::endl;
		}

		VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
		VkResult res = vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
		if (res != VK_SUCCESS) {
			std::cerr << "Failed to allocate single time command buffer, VkResult: " << res 
					  << " (active buffers: " << currentActive << ")" << std::endl;
			throw std::runtime_error("Failed to allocate single time command buffer");
		}

		activeCommandBuffers.fetch_add(1);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		if (res != VK_SUCCESS) {
			std::cerr << "vkBeginCommandBuffer failed in beginSingleTimeCommands for buffer 0x" << std::hex << reinterpret_cast<uint64_t>(commandBuffer) << std::dec << ", VkResult: " << res << std::endl;
			// Still need to free the allocated buffer
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
			activeCommandBuffers.fetch_sub(1);
			throw std::runtime_error("Failed to begin command buffer in beginSingleTimeCommands");
		}

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		if (commandBuffer == VK_NULL_HANDLE) {
			std::cerr << "ERROR: Attempting to end NULL command buffer!" << std::endl;
			throw std::runtime_error("Command buffer is VK_NULL_HANDLE in endSingleTimeCommands");
		}

		int currentActive = activeCommandBuffers.load();
		
		VkResult res = vkEndCommandBuffer(commandBuffer);
		if (res != VK_SUCCESS) {
			std::cerr << "vkEndCommandBuffer failed in endSingleTimeCommands for buffer 0x" 
					  << std::hex << reinterpret_cast<uint64_t>(commandBuffer) << std::dec 
					  << ", VkResult: " << res << std::endl;
			// Even if end fails, we should still try to free the buffer
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
			activeCommandBuffers.fetch_sub(1);
			throw std::runtime_error("Failed to end command buffer in endSingleTimeCommands");
		}

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		res = vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		if (res != VK_SUCCESS) {
			std::cerr << "vkQueueSubmit failed in endSingleTimeCommands for buffer 0x" 
					  << std::hex << reinterpret_cast<uint64_t>(commandBuffer) << std::dec 
					  << ", VkResult: " << res << std::endl;
			throw std::runtime_error("Failed to submit command buffer in endSingleTimeCommands");
		}

		res = vkQueueWaitIdle(graphicsQueue);
		if (res != VK_SUCCESS) {
			std::cerr << "vkQueueWaitIdle failed in endSingleTimeCommands, VkResult: " << res << std::endl;
			throw std::runtime_error("Failed to wait for queue idle in endSingleTimeCommands");
		}

		// Only print debug info when freeing the last buffer
		if (currentActive == 1) {
			std::cout << "Freed last single-time command buffer" << std::endl;
		}
		
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		activeCommandBuffers.fetch_sub(1);
	}

	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;  // Use the parameter, don't override
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = numSamples;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void createDescriptorSets() {
		// resize
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

		// fill in
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorBufferInfo bufferInfo{};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageViews.empty() ? VK_NULL_HANDLE : textureImageViews[0];
			imageInfo.sampler = textureSampler;

			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	void createTextureDescriptorSets() {
		std::cout << "Creating descriptor sets for " << textureImages.size() << " textures..." << std::endl;
		
		// Create one descriptor set per texture
		textureDescriptorSets.resize(textureImages.size());

		// Use the texture descriptor set layout for these sets
		std::vector<VkDescriptorSetLayout> layouts(textureImages.size(), textureDescriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(textureImages.size());
		allocInfo.pSetLayouts = layouts.data();

		VkResult result = vkAllocateDescriptorSets(device, &allocInfo, textureDescriptorSets.data());
		if (result != VK_SUCCESS) {
			std::cerr << "Failed to allocate texture descriptor sets! VkResult: " << result << std::endl;
			std::cerr << "Requested " << textureImages.size() << " descriptor sets" << std::endl;
			throw std::runtime_error("Failed to allocate texture descriptor sets!");
		}

		std::cout << "Successfully allocated " << textureImages.size() << " texture descriptor sets" << std::endl;

		// Configure each descriptor set - ONLY texture sampler, no UBO
		for (size_t i = 0; i < textureImages.size(); i++) {
			VkDescriptorImageInfo imageInfo{};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageViews[i];
			imageInfo.sampler = textureSampler;

			VkWriteDescriptorSet descriptorWrite{};
			descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrite.dstSet = textureDescriptorSets[i];
			descriptorWrite.dstBinding = 0;  // Binding 0 in set 1 (texture layout)
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrite.descriptorCount = 1;
			descriptorWrite.pImageInfo = &imageInfo;

			vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
		}
		
		std::cout << "Successfully configured " << textureImages.size() << " texture descriptor sets" << std::endl;
	}

	void createDescriptorPool() {
		// Calculate how many textures you expect - update this to handle more textures
		const uint32_t MAX_TEXTURES = 50;  // Increased from 10 to 50 to handle large models

		std::array<VkDescriptorPoolSize, 2> poolSizes{};

		// UBO descriptors (one per swap chain image for camera)
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		// Texture sampler descriptors (one per texture + one per swap chain image if you had per-frame textures)
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = MAX_TEXTURES;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size()) + MAX_TEXTURES;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor pool!");
		}
	}

	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	void createDescriptorSetLayout() {
		// Binding 0: UBO (your camera/MVP matrices)
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		uboLayoutBinding.pImmutableSamplers = nullptr;

		// Binding 1: Texture sampler - MAKE SURE YOU HAVE THIS
		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		samplerLayoutBinding.pImmutableSamplers = nullptr;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create descriptor set layout!");
		}

	}

	void createTextureDescriptorSetLayout() {
		// This layout is ONLY for the texture sampler (no UBO)
		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 0;  // Binding 0 in set 1
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		samplerLayoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &samplerLayoutBinding;

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &textureDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create texture descriptor set layout!");
		}
	}

	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		// temp buffer to transfer data from CPU to GPU
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		// actual vertex buffer on the GPU
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
		try {
			commandBuffer = beginSingleTimeCommands();

			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

			endSingleTimeCommands(commandBuffer);
		}
		catch (const std::exception& e) {
			std::cerr << "Error in copyBuffer: " << e.what() << std::endl;
			if (commandBuffer != VK_NULL_HANDLE) {
				vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
				activeCommandBuffers.fetch_sub(1);
			}
			throw;
		}
	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this); // ADD THIS LINE - needed for resize callback
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

		std::cout << "Creating command buffers with pool handle: 0x" << std::hex << reinterpret_cast<uint64_t>(commandPool) << std::dec << std::endl;

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate command buffers!");
		}

		// Debug: print allocated handles
		std::cout << "Allocated command buffers:";
		for (size_t i = 0; i < commandBuffers.size(); ++i) {
			std::cout << " [" << i << "]=0x" << std::hex << reinterpret_cast<uint64_t>(commandBuffers[i]) << std::dec;
		}
		std::cout << std::endl;

		std::cout << "Successfully created " << commandBuffers.size() << " command buffers" << std::endl;
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		if (commandBuffer == VK_NULL_HANDLE) {
			std::cerr << "ERROR: Command buffer is NULL!" << std::endl;
			throw std::runtime_error("Command buffer is VK_NULL_HANDLE!");
		}

		// Sanity check: is this our allocated command buffer?
		if (!commandBuffers.empty()) {
			bool found = false;
			for (size_t i = 0; i < commandBuffers.size(); ++i) {
				if (commandBuffers[i] == commandBuffer) { found = true; break; }
			}
			if (!found) {
				std::cerr << "WARNING: commandBuffer passed to recordCommandBuffer is not in commandBuffers array" << std::endl;
			}
		}

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			std::cerr << "vkBeginCommandBuffer failed for buffer 0x" << std::hex << reinterpret_cast<uint64_t>(commandBuffer) << std::dec << std::endl;
			throw std::runtime_error("Failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkBuffer vertexBuffers[] = { vertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		// IMPORTANT: Bind your camera descriptor set using currentFrame
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipelineLayout, 0, 1,
			&descriptorSets[currentFrame],  // Use currentFrame, not imageIndex!
			0, nullptr);

		// Draw each texture group
		for (const auto& cmd : drawCommands) {
			// Bind the texture descriptor set (binding set 1)
			if (cmd.textureIndex < textureDescriptorSets.size()) {
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
					pipelineLayout, 1, 1,
					&textureDescriptorSets[cmd.textureIndex],
					0, nullptr);

				// Draw this group
				vkCmdDrawIndexed(commandBuffer, cmd.indexCount, 1, cmd.firstIndex, 0, 0);
			}
		}

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			std::cerr << "vkEndCommandBuffer failed for buffer 0x" << std::hex << reinterpret_cast<uint64_t>(commandBuffer) << std::dec << std::endl;
			throw std::runtime_error("Failed to record command buffer!");
		}
	}

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		if (!queueFamilyIndices.graphicsFamily.has_value()) {
			throw std::runtime_error("Graphics queue family not found!");
			}

		//VK_COMMAND_POOL_CREATE_TRANSIENT_BIT for short-lived command buffers
		//VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT to allow resetting individual command buffers
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
		if (result != VK_SUCCESS) {
			std::cerr << "Failed to create command pool, VkResult: " << result << std::endl;
			throw std::runtime_error("failed to create command pool!");
		}

		std::cout << "Command pool created successfully with handle: 0x" << std::hex << reinterpret_cast<uint64_t>(commandPool) << std::dec << std::endl;
	}

	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 3> attachments = {
				colorImageView,
				depthImageView,
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("shaders/vert.spv");
		auto fragShaderCode = readFile("shaders/frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		// Vertex shader stage
		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		// Fragment shader stage
		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		std::array<VkDescriptorSetLayout, 2> setLayouts = {
			descriptorSetLayout,           // Set 0: Camera UBO
			textureDescriptorSetLayout     // Set 1: Texture sampler
		};

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
		pipelineLayoutInfo.pSetLayouts = setLayouts.data();

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP for strip of triangles
		// VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST is the only other triangle option
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		// Viewport setup
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		// Scissor will render to entire viewport
		// Comment for any changes
		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		// Rasterizer setup
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		// VK_POLYGON_MODE_FILL for the default fill mode
		// VK_POLYGON_MODE_LINE for wireframe
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

		rasterizer.lineWidth = 1.0f;

		rasterizer.cullMode = VK_CULL_MODE_NONE;  // Temporarily disable culling to debug missing faces
		// VK_FRONT_FACE_COUNTER_CLOCKWISE for OpenGL
		// VK_FRONT_FACE_CLOCKWISE for the tutorial
		// feel free to change to match a counter clockwise winding order
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;  // Try clockwise first

		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; // Optional
		rasterizer.depthBiasClamp = 0.0f; // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

		// Multisampling setup (disabled for now)
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_TRUE; // enable sample shading in the pipeline
		multisampling.minSampleShading = .2f; // min fraction for sample shading; closer to one is smooth
		multisampling.rasterizationSamples = msaaSamples;
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
		multisampling.alphaToOneEnable = VK_FALSE; // Optional

		// Color blending setup
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

		// Enable alpha blending
		colorBlendAttachment.blendEnable = VK_TRUE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
	 colorBlending.blendConstants[3] = 0.0f; // Optional

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;

		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; // Optional
		pipelineInfo.pColorBlendState = &colorBlending;
			pipelineInfo.pDynamicState = &dynamicState;

		pipelineInfo.layout = pipelineLayout;

			pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;

		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
		pipelineInfo.basePipelineIndex = -1; // Optional

		 VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;

		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds =  0.0f; // Optional
		depthStencil.maxDepthBounds = 1.0f; // Optional

		pipelineInfo.pDepthStencilState = &depthStencil;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
			}

		return shaderModule;
	}

	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	void createRenderPass() {
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

		// VK_ATTACHMENT_LOAD_OP_LOAD to load existing contents
		// VK_ATTACHMENT_LOAD_OP_CLEAR to clear to a constant at the start (default)
		// VK_ATTACHMENT_STORE_OP_DONT_CARE framebuffer contents will be undefined after rendering
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		// VK_IMAGE_LAYOUT_UNDEFINED as we don't care about previous image contents
		// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for presenting to the screen (default)
		// VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL for rendering directly to the image
		colorAttachment.samples = msaaSamples;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = msaaSamples;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;
		subpass.pResolveAttachments = &colorAttachmentResolveRef;
		// pInputAttachments for reading from a framebuffer attachment in a shader
		// pResolveAttachments for multisampling
		// pDepthStencilAttachment for depth and stencil attachments
		// pPreserveAttachments for preserving attachments through a subpass

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (uint32_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
				}
	}

	void cleanupSwapChain() {
		// Clean up color resources
		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);

		// Clean up depth resources
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		// Clean up framebuffers
		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		// DO NOT destroy command buffers here!
		// They are persistent across swap chain recreation

		// Clean up image views
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		// Clean up swap chain
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createColorResources();
		createDepthResources();
		createFramebuffers();
	}

	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily) {
			// concurrent mode for multiple queue families
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			// exclusive mode for single queue family (better performance)
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0; // Optional
			createInfo.pQueueFamilyIndices = nullptr; // Optional
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.sampleRateShading = VK_TRUE; // enable sample shading feature for the device

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		// Swap chain does not work with this code enabled
		//createInfo.enabledExtensionCount = 0;

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) { // early exit if all families are found
				break;
			}

			i++;
		}

		return indices;
	}

	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		// Use an ordered map to automatically sort candidates by increasing score
		std::multimap<int, VkPhysicalDevice> candidates;

		for (const auto& device : devices) {
			int score = isDeviceSuitable(device);
			candidates.insert(std::make_pair(score, device));
		}

		// msaa sample count setup for first suitable device
		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				msaaSamples = getMaxUsableSampleCount();
				break;
			}
		}

		// Check if the best candidate is suitable at all
		if (candidates.rbegin()->first > 0) {
			physicalDevice = candidates.rbegin()->second;
		}
		else {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

	}

	// function for single GPU - returns score instead of bool
	int isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		// Return score instead of bool - suitable devices get a score > 0
		if (indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy) {
			return 1;  // Basic suitable device gets score of 1
		}
		return 0;  // Unsuitable device gets score of 0
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	// VK_PRESENT_MODE_MAILBOX_KHR for triple buffering
	// VK_PRESENT_MODE_FIFO_KHR for moblie support and performance
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;

		createInfo.enabledLayerCount = 0;

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {

		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}

	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		updateUniformBuffer(currentFrame);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
			return; // ADD THIS - don't try to render if we just recreated
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		vkResetFences(device, 1, &inFlightFences[currentFrame]);
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		presentInfo.pResults = nullptr; // Optional

		vkQueuePresentKHR(presentQueue, &presentInfo);

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void updateUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo{};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1; //inversion due to OpenGL

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void cleanup() {
		cleanupSwapChain();

		// Clean up textures
		for (size_t i = 0; i < textureImages.size(); i++) {
			vkDestroyImageView(device, textureImageViews[i], nullptr);
			vkDestroyImage(device, textureImages[i], nullptr);
			vkFreeMemory(device, textureImageMemories[i], nullptr);
		}

		vkDestroySampler(device, textureSampler, nullptr);

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, textureDescriptorSetLayout, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

		vkDestroyRenderPass(device, renderPass, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	std::string normalizePath(const std::string& path) {
		std::string normalized = path;
		// Convert backslashes to forward slashes for consistency
		std::replace(normalized.begin(), normalized.end(), '\\', '/');
		return normalized;
	}

	bool fileExists(const std::string& path) {
		std::ifstream file(path);
		return file.good();
	}

	std::string findTexturePath(const std::string& originalPath) {
		// Normalize the path from MTL file
		std::string normalized = normalizePath(originalPath);
		
		// Try the path as specified in MTL file
		if (fileExists(normalized)) {
			return normalized;
		}
		
		// Try with lowercase textures directory
		std::string lowercasePath = normalized;
		if (lowercasePath.find("Textures/") == 0) {
			lowercasePath.replace(0, 9, "textures/");
			if (fileExists(lowercasePath)) {
				return lowercasePath;
			}
		}
		
		// Try with uppercase Textures directory
		std::string uppercasePath = normalized;
		if (uppercasePath.find("textures/") == 0) {
			uppercasePath.replace(0, 9, "Textures/");
			if (fileExists(uppercasePath)) {
				return uppercasePath;
			}
		}
		
		// Try just the filename with our base texture directory
		size_t lastSlash = normalized.find_last_of('/');
		if (lastSlash != std::string::npos) {
			std::string filename = normalized.substr(lastSlash + 1);
			std::string withBaseDir = TEXTURE_DIR + filename;
			if (fileExists(withBaseDir)) {
				return withBaseDir;
			}
			
			// Try with uppercase directory
			std::string withUpperDir = "Textures/" + filename;
			if (fileExists(withUpperDir)) {
				return withUpperDir;
			}
		}
		
		// If all else fails, return the original normalized path
		return normalized;
	}

	void debugFileStructure() {
		std::cout << "\n=== FILE STRUCTURE DEBUG ===" << std::endl;
		
		// Check if model file exists
		std::cout << "Model file check:" << std::endl;
		std::cout << "  " << MODEL_PATH << " exists: " << (fileExists(MODEL_PATH) ? "YES" : "NO") << std::endl;
		
		// Check texture directories
		std::cout << "\nTexture directory checks:" << std::endl;
		std::cout << "  textures/ directory accessible: " << (fileExists("textures/") ? "YES" : "NO") << std::endl;
		std::cout << "  Textures/ directory accessible: " << (fileExists("Textures/") ? "YES" : "NO") << std::endl;
		
		// Check specific texture files mentioned in TEXTURE_PATHS
		std::cout << "\nDirect texture file checks:" << std::endl;
		for (const auto& texPath : TEXTURE_PATHS) {
			std::cout << "  " << texPath << " exists: " << (fileExists(texPath) ? "YES" : "NO") << std::endl;
		}
		
		// Check texture files from MTL file
		std::vector<std::string> mtlTextures = {
			"Textures/Engines_WingsColor.tga",
			"textures/Engines_WingsColor.tga", 
			"Textures/Fuselage_CockpitColor.tga",
			"textures/Fuselage_CockpitColor.tga"
		};
		
		std::cout << "\nMTL texture file checks:" << std::endl;
		for (const auto& texPath : mtlTextures) {
			std::cout << "  " << texPath << " exists: " << (fileExists(texPath) ? "YES" : "NO") << std::endl;
		}
		
		std::cout << "========================\n" << std::endl;
	}
};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}