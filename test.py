'''arr1 = [100, 20, 300, 400,50,66,11,2]
arr2 = [1, 25, 150]

# First, sort both individually
arr1.sort()
arr2.sort()

i = 0
j = 0
arr3 = []

# Compare values from both arrays
while i < len(arr1) and j < len(arr2):
    if arr1[i] < arr2[j]:
        arr3.append(arr1[i])
        i += 1
    else:
        arr3.append(arr2[j])
        j += 1

# Add remaining elements
while i < len(arr1):
    arr3.append(arr1[i])
    i += 1

while j < len(arr2):
    arr3.append(arr2[j])
    j += 1

print(arr3)
'''
'''names = ["Alex", "Priya", "Jordan", "Meera", "David"]
ages = [21, 25, 30, 18, 27]
name= "Alex"
class Name:
    def __init__(self,names,ages):
        self.names = names
        self.ages = ages
        
    def getAge(self,name):
        if name in self.names:
            index = self.names.index(name)
            return self.ages[index]
        else:
            return None

obj = Name(names, ages)
print(obj.getAge("Alex"))   # 21
print(obj.getAge("Meera"))  # 18
print(obj.getAge("Sam"))    # N
        '''

'''class parent:
    def __init__(self,name, age, school):
        self.name = name
        self.age = age
        self.school = school
    def show(self):
        print(f"My name is {self.name} and i am {self.age}, i go to {self.school}")
class child(parent):
    def grades(self,subject1, subject2):
        self.subject1 = subject1
        self.subject2 = subject2

    def averageMarks(self):
        return (self.subject1 + self.subject2) / 2

c = child("alex",19,"Pillai college")
c.show()
c.grades(19,20)
print("averahe marks", c.averageMarks())        '''
'''string = "i am anish"
rev_str = " ".join(string.split()[::-1])

print(rev_str)'''
''''class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def appendnode(self,data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr=curr.next
        curr.next = new_node
    def showList(self):
        curr = self.head
        while curr:
            print(curr.data)
            curr = curr.next
        print("None")

l1 = LinkedList()
l1.appendnode(1)
l1.appendnode(2)
l1.appendnode(3)
l1.appendnode(4)
l1.appendnode(5)
l1.showList()'''
'''class myarr:
    def __init__(self ):
        self.nums1 = []
    def addNums(self,value):
        self.nums1.append(value)
        return self.nums1
    def showNums(self):
        print(self.nums1)
    def removeDuplicate(self):
        seen = []
        for num in self.nums1:
            if num not in seen:
                seen.append(num)
        self.nums1 = seen'''

'''n = myarr()
n.addNums(1)
n.addNums(2)
n.addNums(2)
n.addNums(3)
n.removeDuplicate()
n.showNums()
'''
'''def firstUniqChar(s):
    # loop through each character
    for i in range(len(s)):
        unique = True   # assume it's unique
        for j in range(len(s)):
            if i != j and s[i] == s[j]:  # found duplicate
                unique = False
                break
        if unique:
            return i   # first unique char index
    return -1   # if none found

# Example
print(firstUniqChar("leetcode"))      # 0  (l is unique)
print(firstUniqChar("loveleetcode"))  # 2  (v is first unique)
print(firstUniqChar("aabb"))          # -1 (no unique)'''
'''import re
s = "racecar"
cleaned = re.sub(r'[^a-z0-9]', '', s.lower())
rev_s = str(cleaned[::-1])
if (rev_s == cleaned):
    print(f"{s} is Palindrome")
else:
    print(f"{s} is not Paindrome")
'''
'''
class CheckReverse:
    def __init__(self, s):
        self.s= s
    def ReverseString(self):
        rev_s = "".join(reversed(self.s))
        print(rev_s)
        if (rev_s == self.s):
            return True
        else:
            return False

s = CheckReverse("anish")
print(s.ReverseString())'''

'''
class WordCount:
    def __init__(self,word):
        self.word = word
    def count_words(self):
        count = self.word.split()
        return len(count)
    def longest_word(self):
        words = self.word.split()
        return max(words, key=len)

wc = WordCount("Python is really powerful")
print(wc.count_words())'''
'''import re
text = "my mail is "
result = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",text)
print(result)'''
'''

class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList:
    def __init__(self):
        self.head = None
    def appendnode(self,data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr=curr.next
        curr.next = new_node
    
    def first_repeating(head):
        seen = set()
        while head:
            if head.data in seen:
                return head.data  # first repeating value
            seen.add(head.data)
            head = head.next
        return None 
   

l1 = LinkedList()

l1.appendnode(1)
l1.appendnode(1)
l1.appendnode(3)
l1.appendnode(4)
l1.appendnode(5)

l1.first_repeating()'''
'''
class Node:
    def __init__(self,data):
        self.data = data
        self.next= None
class LinkedList:
    def __init__(self):
        self.head = None
    def appendnode(self,data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next= new_node
    def showList(self):
        curr = self.head
        while curr:
            print(curr.data)
            curr=curr.next
        print("None")
    def checkReverse(self):
        seen = set()
        curr = self.head
        while curr:
            if curr.data in seen:
                return curr.data
            seen.add(curr.data)
            curr = curr.next
    def checkSequence(self):
        seen = set()
        repeats = set()
        curr = self.head
        while curr:
            if curr.data in seen:
                repeats.add(curr.data)
            else:
                seen.add(curr.data)
            curr = curr.next
        return list(repeats)
    def reverseList(self):
       prev = None
       curr = self.head
       while curr:
           next_node = curr.next
           curr.next = prev
           prev = curr
           curr = next_node
       self.head = prev
       return self.head
        

        


l1 = LinkedList()

l1.appendnode(1)
l1.appendnode(1)
l1.appendnode(3)
l1.appendnode(4)
l1.appendnode(5)
l1.showList()
l1.checkReverse()
print(l1.checkReverse())
print(l1.checkSequence()) 
print(l1.reverseList())
l1.showList()'''
'''s = "hello"
t = "heelo"
if sorted(s) == sorted(t):
    print("anagram")
else:
    print("not anagram")'''
'''s = "{[]}"
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {")":"(", "}":"{", "]":"["}
        for char in s:
            if char in mapping:
                top_element = stack.pop() if stack else '#'
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
        return not stack
print(Solution().isValid(s))'''

'''s = -121
x = -1 if s < 0 else 1   # x = -1 for negative, 1 for positive
rev_s = int(str(abs(s))[::-1]) * x  # reverse the digits and restore sign
if rev_s ==s:
    print(f"{s} is palindrome")
else:
    print(f"{s} is not palindrome")'''

'''arr = [5,9,3,7,8]
for i in range(len(arr)//2):
    for j in range(len(arr)-1):
        if arr[i] < arr[j]:
            temp = arr[i]
            arr[i]= arr[j]
            arr[j] = temp
print(arr)
'''
'''class Node():
    def __init__(self,data):
        self.data = data
        self.next = None
class LinkedList():
    def __init__(self):
        self.head = None
    def appendnode(self,data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node
    def showList(self):
        curr = self.head
        while curr:
            print(curr.data)
            curr = curr.next
        print("None")
    def checkDuplicate(self):
        seen = set()
        repeats = set()
        curr = self.head
        while curr:
            if curr in seen:
                repeats.add(curr.data)
            else:
                seen.add(curr.data)
            curr= curr.next


l1 = LinkedList()

l1.appendnode(1)
l1.appendnode(1)
l1.appendnode(3)
l1.appendnode(4)
l1.appendnode(5)
l1.showList()'''
s = "abbcc"
arr = list(s)
seen = set()
twins = set()
for ch in arr:
    if ch in seen:
        twins.add(ch)
    else:
        seen.add(ch)
print(seen)
print(twins)

