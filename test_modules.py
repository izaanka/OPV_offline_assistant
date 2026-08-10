"""Test suite for OPV Voice Assistant Module System."""

import os
import unittest
from unittest.mock import patch

import modules_registry


class TestModuleSystem(unittest.TestCase):

    def setUp(self):
        self.modules = modules_registry.load_modules()

    def test_loaded_modules(self):
        expected_modules = ["website_reader", "file_manager", "terminal", "weather", "web_search"]
        for mod in expected_modules:
            self.assertIn(mod, self.modules, f"Module '{mod}' should be registered.")

    def test_file_manager_operations(self):
        fm = self.modules["file_manager"]
        test_file = "test_sample_unit.txt"
        if os.path.exists(test_file):
            os.remove(test_file)

        # Create
        res_create = fm.execute({"action": "create", "path": test_file, "content": "Unit Test 123"})
        self.assertIn("successfully created", res_create.lower())
        self.assertTrue(os.path.exists(test_file))


        # Read
        res_read = fm.execute({"action": "read", "path": test_file})
        self.assertIn("Unit Test 123", res_read)

        # Delete (with mock confirmation returning True)
        with patch("modules_registry.confirm_action", return_value=True):
            res_del = fm.execute({"action": "delete", "path": test_file})
            self.assertIn("successfully deleted", res_del.lower())
            self.assertFalse(os.path.exists(test_file))

    def test_terminal_module(self):
        term = self.modules["terminal"]
        with patch("modules_registry.confirm_action", return_value=True):
            res = term.execute({"command": "echo 'Hello OPV Modules'"})
            self.assertIn("Hello OPV Modules", res)

    def test_website_reader_module(self):
        wr = self.modules["website_reader"]
        res = wr.execute({"url": "https://example.com"})
        self.assertIn("content fetched", res.lower())
        self.assertIn("example domain", res.lower())


    def test_bbc_query_direct_context(self):
        query = "What is currently on the BBC website homepage?"
        ctx = modules_registry.get_direct_context(query)
        self.assertIsNotNone(ctx)
        self.assertIn("website_reader", ctx)
        self.assertIn("bbc.com", ctx.lower())


    def test_dynamic_extensibility(self):
        # Create a new module file dynamically
        dummy_path = os.path.join(os.path.dirname(__file__), "modules", "dummy_test_plugin.py")
        dummy_code = '''
from modules_registry import BaseModule

class DummyPlugin(BaseModule):
    name = "dummy_plugin"
    description = "A dummy plugin for testing zero-code-change extensibility"

    def execute(self, params, user_input=""):
        return "Dummy plugin executed!"
'''
        with open(dummy_path, "w") as f:
            f.write(dummy_code)

        try:
            # Reload modules
            reloaded = modules_registry.load_modules()
            self.assertIn("dummy_plugin", reloaded)
            res = modules_registry.execute_module("dummy_plugin", {})
            self.assertEqual(res, "Dummy plugin executed!")
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
            modules_registry.load_modules()


if __name__ == "__main__":
    unittest.main()
