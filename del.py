class Node:
    def __init__(self,value):
        self.value = value
        self.next = None
class Queue:
    def __init__(self):
        self.rear = None
        self.front = None
        self.size = 0
    
    def enqueue(self, value):
        newNode = Node(value)
        if self.rear == None:
            self.rear = self.front = newNode
            self.length += 1
            return
        self.rear.next = newNode
        self.rear = newNode
        self.length += 1

    def dequeue(self):
        if self.isEmpty():
            return "the queue is empty"
        temp = self.front
        self.front = temp.next
        self.size -=1
        if self.front == None:
            self.rear = None
        return temp.value

    
    def peek(self):
        if self.isEmpty():
            return "queue is empty"
        return self.front.value
    
    def isEmpty(self):
        return (self.size == 0)
    
    def queueSize(self):
        return self.size
    
    def traverseAndPrint(self):
        currentNode = self.front
        while currentNode:
            print(currentNode.value, end = "->")
            currentNode = currentNode.next
        print()
        
x=Queue()
x.enqueue(1)
x.enqueue(2)
x.enqueue(3)
x.enqueue(4)

x.traverseAndPrint()

print(x.queueSize())
print(x.isEmpty())
print(x.peek())
print(x.dequeue())
x.traverseAndPrint()


