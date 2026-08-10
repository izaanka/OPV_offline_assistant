"""Unit tests for ChatGUI module."""

import unittest
from chat_gui import ChatGUI


class TestChatGUI(unittest.TestCase):

    def test_gui_queue_posting(self):
        gui = ChatGUI(assistant_name="TestAssistant")
        gui.post_user_message("Hello AI")
        gui.post_ai_message("Hello User")
        gui.post_system_message("System Notice")
        gui.update_status("Ready", "#a6e3a1")

        self.assertFalse(gui.msg_queue.empty())
        item1 = gui.msg_queue.get()
        self.assertEqual(item1, ("user", ("Hello AI",)))

        item2 = gui.msg_queue.get()
        self.assertEqual(item2, ("ai", ("Hello User",)))

        item3 = gui.msg_queue.get()
        self.assertEqual(item3, ("system", ("System Notice",)))

        item4 = gui.msg_queue.get()
        self.assertEqual(item4, ("status", ("Ready", "#a6e3a1")))


if __name__ == "__main__":
    unittest.main()
