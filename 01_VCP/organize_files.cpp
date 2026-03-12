/**
 * File Organizer for VCP Project
 * 
 * This program organizes:
 * 1. Preprocessed data (thickness/orientation maps) from DRIVE_training/ and IOSTAR_training/ to separate folders
 * 2. Result PNG files from results/ to categorized folders in results/logs/
 * 
 * Compile: g++ -o organize_files organize_files.cpp -std=c++17
 * Run: ./organize_files
 */

#include <iostream>
#include <filesystem>
#include <string>
#include <regex>
#include <map>
#include <vector>

namespace fs = std::filesystem;

// Configuration - Uses relative paths from project root
const std::string PROCESSED_DATA_SOURCES[] = {
    "processed_data/DRIVE_training",
    "processed_data/IOSTAR_training"
};
const std::string THICKNESS_DST = "processed_data/thickness_maps";
const std::string ORIENTATION_DST = "processed_data/orientation_maps";
const std::string RESULTS_SOURCES[] = {
    "results",
    "results/DRIVE",
    "results/IOSTAR"
};
const std::string LOGS_DST = "results/logs";

/**
 * Move a file to destination directory (cut & paste)
 */
bool moveFile(const fs::path& src, const fs::path& dstDir) {
    try {
        fs::path dst = dstDir / src.filename();
        // If destination exists, remove it first
        if (fs::exists(dst)) {
            fs::remove(dst);
        }
        fs::rename(src, dst);
        return true;
    } catch (const fs::filesystem_error& e) {
        // If rename fails (cross-device), try copy + remove
        try {
            fs::path dst = dstDir / src.filename();
            fs::copy(src, dst, fs::copy_options::overwrite_existing);
            fs::remove(src);
            return true;
        } catch (const fs::filesystem_error& e2) {
            std::cerr << "Error moving " << src << ": " << e2.what() << std::endl;
            return false;
        }
    }
}

/**
 * Organize preprocessed thickness and orientation maps
 */
void organizePreprocessedData() {
    std::cout << "\n=== Organizing Preprocessed Data ===" << std::endl;
    
    int thicknessCount = 0;
    int orientationCount = 0;
    
    // Create destination directories if they don't exist
    fs::create_directories(THICKNESS_DST);
    fs::create_directories(ORIENTATION_DST);
    
    // Process each source directory
    for (const auto& srcDir : PROCESSED_DATA_SOURCES) {
        if (!fs::exists(srcDir)) {
            std::cout << "Source directory not found (skipping): " << srcDir << std::endl;
            continue;
        }
        
        std::cout << "Processing: " << srcDir << std::endl;
        
        for (const auto& entry : fs::directory_iterator(srcDir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            
            // Check file type and move to appropriate folder
            if (filename.find("_thickness") != std::string::npos) {
                if (moveFile(entry.path(), THICKNESS_DST)) {
                    thicknessCount++;
                }
            }
            else if (filename.find("_orientation") != std::string::npos) {
                if (moveFile(entry.path(), ORIENTATION_DST)) {
                    orientationCount++;
                }
            }
        }
    }
    
    std::cout << "Thickness files moved: " << thicknessCount << std::endl;
    std::cout << "Orientation files moved: " << orientationCount << std::endl;
}

/**
 * Organize result PNG files into categorized folders
 */
void organizeResultFiles() {
    std::cout << "\n=== Organizing Result Files ===" << std::endl;
    
    // Define mapping from filename pattern to destination folder
    std::map<std::string, std::string> patternToFolder = {
        {"_od.png", "od_png"},
        {"_orientation.png", "orientation_png"},
        {"_av_pixelwise.png", "pixelwise_png"},
        {"_thickness.png", "thickness_png"},
        {"_topology.png", "topology_png"},
        {"_av_treewise.png", "treewise_png"},
        {"_vessel.png", "vessel_png"}
    };
    
    // Count for each category
    std::map<std::string, int> counts;
    for (const auto& [pattern, folder] : patternToFolder) {
        counts[folder] = 0;
        // Create destination folders if they don't exist
        fs::create_directories(LOGS_DST + "/" + folder);
    }
    
    // Process each results source directory
    for (const auto& srcDir : RESULTS_SOURCES) {
        if (!fs::exists(srcDir)) {
            continue;
        }
        
        for (const auto& entry : fs::directory_iterator(srcDir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            
            // Skip non-PNG files
            if (entry.path().extension() != ".png") continue;
            
            // Match pattern and move to appropriate folder
            for (const auto& [pattern, folder] : patternToFolder) {
                if (filename.find(pattern) != std::string::npos) {
                    fs::path dstDir = fs::path(LOGS_DST) / folder;
                    if (moveFile(entry.path(), dstDir)) {
                        counts[folder]++;
                    }
                    break;  // File matched, no need to check other patterns
                }
            }
        }
    }
    
    // Print summary
    std::cout << "\nFiles organized by category:" << std::endl;
    for (const auto& [folder, count] : counts) {
        std::cout << "  " << folder << ": " << count << " files" << std::endl;
    }
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "VCP Project File Organizer" << std::endl;
    std::cout << "============================================" << std::endl;
    
    // 1. Organize preprocessed data
    organizePreprocessedData();
    
    // 2. Organize result files
    organizeResultFiles();
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "File organization complete!" << std::endl;
    std::cout << "============================================" << std::endl;
    
    return 0;
}
