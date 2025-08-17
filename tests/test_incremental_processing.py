import pytest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import hashlib

# Add the parent directory to the path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.change_tracker import ChangeTracker
from src.config import Config
from src.extractor import DocumentExtractor


class TestIncrementalProcessing:
    """Comprehensive test suite for incremental processing functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_data_dir(self, temp_dir):
        """Create test data directory with sample files"""
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        
        # Create sample test files
        (data_dir / "file1.txt").write_text("This is the content of file 1. It has multiple sentences.")
        (data_dir / "file2.txt").write_text("This is file 2 content. Different from file 1.")
        (data_dir / "file3.txt").write_text("File 3 has its own content. Unique and different.")
        
        return data_dir
    
    @pytest.fixture
    def test_cache_dir(self, temp_dir):
        """Create test cache directory"""
        cache_dir = temp_dir / ".pipeline_cache"
        return cache_dir
    
    @pytest.fixture
    def mock_config(self, test_data_dir, temp_dir):
        """Create mock config for testing"""
        config = MagicMock(spec=Config)
        config.data_dir = str(test_data_dir)
        config.output_dir = temp_dir / "output"
        config.output_dir.mkdir()
        config.tokenizer = "tiktoken:cl100k_base"
        return config
    
    @pytest.fixture
    def change_tracker(self, test_data_dir, test_cache_dir):
        """Create change tracker instance"""
        return ChangeTracker(test_data_dir, test_cache_dir)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Helper to calculate SHA1 hash like the change tracker"""
        hash_sha1 = hashlib.sha1()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha1.update(chunk)
        return hash_sha1.hexdigest()
    
    def test_no_changes_skip_processing(self, change_tracker, test_data_dir):
        """Test 1: No changes - should skip pipeline entirely"""
        # First run - establish baseline
        files_to_process = change_tracker.get_files_to_process()
        assert len(files_to_process) == 3  # All files are "new"
        
        # Update manifest as if processed
        change_tracker.update_manifest(files_to_process)
        
        # Second run - no changes
        assert change_tracker.should_process() == False
        files_to_process = change_tracker.get_files_to_process()
        assert len(files_to_process) == 0
    
    def test_one_new_file_processed(self, change_tracker, test_data_dir):
        """Test 2: One new file - should process only the new file"""
        # Establish baseline
        initial_files = change_tracker.get_files_to_process()
        change_tracker.update_manifest(initial_files)
        
        # Add new file
        new_file = test_data_dir / "new_file.txt"
        new_file.write_text("This is a brand new file with new content.")
        
        # Check detection
        assert change_tracker.should_process() == True
        files_to_process = change_tracker.get_files_to_process()
        assert len(files_to_process) == 1
        assert files_to_process[0].name == "new_file.txt"
    
    def test_one_modified_file_processed(self, change_tracker, test_data_dir):
        """Test 3: One modified file - should process only the modified file"""
        # Establish baseline
        initial_files = change_tracker.get_files_to_process()
        change_tracker.update_manifest(initial_files)
        
        # Modify existing file
        file_to_modify = test_data_dir / "file1.txt"
        original_content = file_to_modify.read_text()
        file_to_modify.write_text(original_content + " Additional content added.")
        
        # Check detection
        assert change_tracker.should_process() == True
        files_to_process = change_tracker.get_files_to_process()
        assert len(files_to_process) == 1
        assert files_to_process[0].name == "file1.txt"
    
    def test_multiple_changes_processed(self, change_tracker, test_data_dir):
        """Test 4: Multiple changes (1 new + 2 modified) - should process exactly these 3"""
        # Establish baseline
        initial_files = change_tracker.get_files_to_process()
        change_tracker.update_manifest(initial_files)
        
        # Add new file
        new_file = test_data_dir / "new_file.txt"
        new_file.write_text("Brand new file content.")
        
        # Modify two existing files
        file1 = test_data_dir / "file1.txt"
        file2 = test_data_dir / "file2.txt"
        file1.write_text(file1.read_text() + " Modified content 1.")
        file2.write_text(file2.read_text() + " Modified content 2.")
        
        # Check detection
        assert change_tracker.should_process() == True
        files_to_process = change_tracker.get_files_to_process()
        assert len(files_to_process) == 3
        
        processed_names = {f.name for f in files_to_process}
        expected_names = {"new_file.txt", "file1.txt", "file2.txt"}
        assert processed_names == expected_names
    
    def test_deleted_file_handled(self, change_tracker, test_data_dir):
        """Test: Deleted file detection"""
        # Establish baseline
        initial_files = change_tracker.get_files_to_process()
        change_tracker.update_manifest(initial_files)
        
        # Delete a file
        file_to_delete = test_data_dir / "file3.txt"
        file_to_delete.unlink()
        
        # Check detection
        new_files, modified_files, deleted_files = change_tracker.detect_changes()
        assert len(deleted_files) == 1
        assert deleted_files[0].name == "file3.txt"
    
    def test_reset_clears_all_cache(self, change_tracker, test_cache_dir, temp_dir):
        """Test: Reset flag clears all cache and storage"""
        # Create some cache data
        change_tracker.cache_intermediate_result("test_step", {"data": "test"})
        manifest = {"file1.txt": {"hash": "testhash"}}
        change_tracker.save_manifest(manifest)
        
        # Create output and lightrag directories
        output_dir = temp_dir / "output"
        lightrag_dir = temp_dir / "lightrag_storage"
        output_dir.mkdir()
        lightrag_dir.mkdir()
        
        # Verify they exist
        assert test_cache_dir.exists()
        assert output_dir.exists()
        assert lightrag_dir.exists()
        
        # Reset
        change_tracker.reset_cache()
        
        # Verify they're gone
        assert not test_cache_dir.exists()
        assert not output_dir.exists()
        assert not lightrag_dir.exists()
    
    def test_manifest_updated_after_processing(self, change_tracker, test_data_dir):
        """Test: Manifest is correctly updated after processing"""
        files_to_process = change_tracker.get_files_to_process()
        
        # Update manifest
        change_tracker.update_manifest(files_to_process)
        
        # Load and verify manifest
        manifest = change_tracker.load_manifest()
        assert len(manifest) == 3
        
        for file_path in files_to_process:
            file_str = str(file_path)
            assert file_str in manifest
            assert "hash" in manifest[file_str]
            assert "mtime" in manifest[file_str]
            assert "last_processed" in manifest[file_str]
            
            # Verify hash is correct
            expected_hash = self.calculate_file_hash(file_path)
            assert manifest[file_str]["hash"] == expected_hash
    
    def test_cache_directory_structure(self, change_tracker):
        """Test: Cache directories are created properly"""
        # Directories should be created during initialization
        assert change_tracker.cache_dir.exists()
        assert change_tracker.intermediate_dir.exists()
        assert change_tracker.manifest_file.parent.exists()
    
    def test_file_hash_consistency(self, test_data_dir):
        """Test: File hashing is consistent and uses SHA1"""
        file_path = test_data_dir / "file1.txt"
        
        # Calculate hash multiple times
        hash1 = self.calculate_file_hash(file_path)
        hash2 = self.calculate_file_hash(file_path)
        
        # Should be identical
        assert hash1 == hash2
        
        # Should be SHA1 length (40 hex chars)
        assert len(hash1) == 40
        assert all(c in "0123456789abcdef" for c in hash1)
    
    @patch('src.extractor.DocumentExtractor.extract_documents')
    def test_extractor_with_specific_files(self, mock_extract, mock_config, test_data_dir):
        """Test: Extractor processes only specified files"""
        mock_extract.return_value = []
        
        extractor = DocumentExtractor(mock_config)
        
        # Test with specific files
        specific_files = [test_data_dir / "file1.txt", test_data_dir / "file2.txt"]
        extractor.run(specific_files)
        
        # Verify extract_documents was called with specific files
        mock_extract.assert_called_once_with(specific_files)
    
    @patch('src.extractor.DocumentExtractor.extract_documents')
    def test_extractor_with_no_files_specified(self, mock_extract, mock_config):
        """Test: Extractor processes all files when none specified"""
        mock_extract.return_value = []
        
        extractor = DocumentExtractor(mock_config)
        extractor.run()
        
        # Verify extract_documents was called with None (all files)
        mock_extract.assert_called_once_with(None)
    
    def test_status_summary_accuracy(self, change_tracker, test_data_dir):
        """Test: Status summary provides accurate counts"""
        # Initial state - all files are new
        status = change_tracker.get_status_summary()
        assert status["total_files"] == 3
        assert status["new_files"] == 3
        assert status["modified_files"] == 0
        assert status["deleted_files"] == 0
        assert status["files_to_process"] == 3
        
        # After processing
        files_to_process = change_tracker.get_files_to_process()
        change_tracker.update_manifest(files_to_process)
        
        status = change_tracker.get_status_summary()
        assert status["total_files"] == 3
        assert status["new_files"] == 0
        assert status["modified_files"] == 0
        assert status["deleted_files"] == 0
        assert status["files_to_process"] == 0
        
        # After adding new file and modifying existing
        new_file = test_data_dir / "new_file.txt"
        new_file.write_text("New content")
        (test_data_dir / "file1.txt").write_text("Modified content")
        
        status = change_tracker.get_status_summary()
        assert status["total_files"] == 4
        assert status["new_files"] == 1
        assert status["modified_files"] == 1
        assert status["deleted_files"] == 0
        assert status["files_to_process"] == 2
    
    def test_cache_intermediate_results(self, change_tracker):
        """Test: Intermediate results caching works correctly"""
        # Test list data (saves as JSONL)
        list_data = [{"id": 1, "text": "test"}, {"id": 2, "text": "test2"}]
        change_tracker.cache_intermediate_result("test_list", list_data)
        
        # Test dict data (saves as JSON)
        dict_data = {"key": "value", "number": 42}
        change_tracker.cache_intermediate_result("test_dict", dict_data)
        
        # Verify caching
        assert change_tracker.has_cached_result("test_list")
        assert change_tracker.has_cached_result("test_dict")
        assert not change_tracker.has_cached_result("nonexistent")
        
        # Verify loading
        loaded_list = change_tracker.load_cached_result("test_list")
        loaded_dict = change_tracker.load_cached_result("test_dict")
        
        assert loaded_list == list_data
        assert loaded_dict == dict_data
        assert change_tracker.load_cached_result("nonexistent") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])