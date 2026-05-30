from queue import Queue, Empty
from typing import Any, Dict, List, Optional
import time
import threading


class Message:
    def __init__(self, sender: str, receiver: str, msg_type: str, content: Any = None):
        self.sender = sender
        self.receiver = receiver
        self.msg_type = msg_type   # info, ack, reject, error, damage
        self.content = content
        self.timestamp = time.time()
        self.status = "new"

    def __repr__(self):
        return f"Message(from={self.sender}, to={self.receiver}, type={self.msg_type}, content={self.content}, status={self.status})"


class MessageBroker:
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.lock = threading.Lock()

    def register_agent(self, agent_id: str):
        with self.lock:
            if agent_id not in self.queues:
                self.queues[agent_id] = Queue()

    def send_message(self, message: Message):
        with self.lock:
            if message.receiver in self.queues:
                self.queues[message.receiver].put(message)
            else:
                raise ValueError(f"Хүлээн авагч олдсонгүй: {message.receiver}")

    def get_message(self, agent_id: str, timeout: Optional[float] = None) -> Message:
        if agent_id in self.queues:
            return self.queues[agent_id].get(timeout=timeout)
        raise ValueError(f"Агент олдсонгүй: {agent_id}")


class Agent:
    def __init__(self, agent_id: str, broker: MessageBroker):
        self.agent_id = agent_id
        self.broker = broker
        self.broker.register_agent(agent_id)
        self.received_messages: List[Message] = []
        self._running = False
        self._thread = None

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._process_messages, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def send_message(self, receiver: str, msg_type: str, content: Any = None):
        message = Message(self.agent_id, receiver, msg_type, content)
        self.broker.send_message(message)

    def _process_messages(self):
        while self._running:
            try:
                message = self.broker.get_message(self.agent_id, timeout=0.5)
                self.received_messages.append(message)
                self._handle_message(message)
            except Empty:
                continue
            except Exception as e:
                print(f"Алдаа гарлаа {self.agent_id}: {str(e)}")

    def _handle_message(self, message: Message):
        print(f"[{self.agent_id}] хүлээн авсан: {message}")

        if message.msg_type == "info":
            print(f"{self.agent_id}: мэдээлэл авлаа -> {message.content}")
            self.send_message(message.sender, "ack", "Мэдээлэл хүлээн авлаа")

        elif message.msg_type == "ack":
            print(f"{self.agent_id}: баталгаажуулалт авлаа -> {message.content}")

        elif message.msg_type == "reject":
            print(f"{self.agent_id}: татгалзсан хариу авлаа -> {message.content}")

        elif message.msg_type == "error":
            print(f"{self.agent_id}: алдаа -> {message.content}")

        elif message.msg_type == "damage":
            print(f"{self.agent_id}: damage мэдээлэл -> {message.content}")

        else:
            print(f"{self.agent_id}: үл мэдэгдэх message type")
            self.send_message(message.sender, "error", "Unknown message type")


def main():
    broker = MessageBroker()

    agent1 = Agent("agent1", broker)
    agent2 = Agent("agent2", broker)

    agent1.start()
    agent2.start()

    agent1.send_message("agent2", "info", {"own_loc": (1, 2), "opp_loc": (5, 6)})
    time.sleep(1)

    agent2.send_message("agent1", "damage", {"amount": 25})
    time.sleep(1)

    agent1.stop()
    agent2.stop()


if __name__ == "__main__":
    main()