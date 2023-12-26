class Node:
    """
    Node of an double linked list
    """
    def __init__(self, data):
        self.data = data
        self.next_forward = None
        self.next_backward = None

class DLListLimit:
    """
    Double linked list used as n memory buffer for frame forwarding
        -> limit is the max number of frames in the list
        -> offset is the offset of the read head
        = 0 for the middle of the list
        < 0 to push the read head to the head of the list
        > 0 to push the read head to the tail of the list

    """
    def __init__(self, limit, offset):
        self.head = None
        self.read = None
        self.tail = None
        self.limit = limit
        self.offset = offset
        self.half_limit = int(self.limit / 2) + self.offset
        self.index = 0
        self.is_full = False
        self.is_half = False

    """
    Add one frame or data in the front of the list and move the read head if is necesary
    :arg data to and in the front of the list
    :returns False in case of an full list and true in case of success adding
    """
    def __add_front__(self, data):
        if self.index >= self.limit:
            self.is_full = True
            return False
            
        new_node = Node(data)
        self.index += 1
        if self.head is None:
            self.head = new_node
            self.tail = self.head
            return True
        else:
            self.head.next_backward = new_node
            new_node.next_forward = self.head
            self.head = new_node
            
            if self.index == self.half_limit:
              self.read = self.tail.next_backward
              self.is_half = True
            
            if self.index > self.half_limit:
              self.read = self.read.next_backward
            
        return True

    """
    Delet the tail node of the list
    """
    def __del_end_node__(self):
      tmp = self.tail
      self.tail = tmp.next_backward
      tmp.data = None
      tmp.next_forward = None
      tmp.next_backward = None
      self.is_full = False
      self.index -= 1
      return

    """
    Add an element in the front of the list, 
    test if the addition is successfuly made
    and if not will delete the tail of the
    list and will try second time to append the data 
    """
    def add_to_front(self, data):
      if self.__add_front__(data) is False:
        self.__del_end_node__()
        self.__add_front__(data)
      return


    """
    Print the all list content
    """
    def print_all(self):
        tmp = self.head
        for x in range(0, self.index, 1):
            tmp = tmp.next_forward
        return

    """
    Read data from the list -> read head of the list
    :returns data
    """
    def read_data(self):
      return self.read.data
