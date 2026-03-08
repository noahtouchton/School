#pragma once
#include <string>

struct Node {
    int data;
    Node* next;

    Node(int value) : data(value), next(nullptr) {}
};

class LinkedList {
    private:
        Node* head;
        int size;

    public:
        LinkedList() : head(nullptr), size(0) {}

        void add(int value) {
            Node* newNode = new Node(value);
            if (!head) {
                head = newNode;
            } else {
                Node* current = head;
                while (current->next) {
                    current = current->next;
                }
                current->next = newNode;
            }
            size++;
        }

        void remove(int value) {
            if (!head) return;

            if (head->data == value) {
                Node* temp = head;
                head = head->next;
                delete temp;
                size--;
                return;
            }

            Node* current = head;
            while (current->next && current->next->data != value) {
                current = current->next;
            }

            if (current->next) {
                Node* temp = current->next;
                current->next = current->next->next;
                delete temp;
                size--;
            }
        }

        int pop() {
            if (!head) throw std::runtime_error("List is empty");
            Node* temp = head;
            int value = head->data;
            head = head->next;
            delete temp;
            size--;
            return value;
        }

        bool contains(int value) {
            Node* current = head;
            while (current) {
                if (current->data == value) return true;
                current = current->next;
            }
            return false;
        }

        int getSize() {
            return size;
        }

        void shuffle() {
            if (size < 2) return;

            // Convert linked list to array for easy shuffling
            int* arr = new int[size];
            Node* current = head;
            for (int i = 0; i < size; i++) {
                arr[i] = current->data;
                current = current->next;
            }

            // Shuffle the array using Fisher-Yates algorithm
            for (int i = size - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                std::swap(arr[i], arr[j]);
            }

            // Convert back to linked list
            current = head;
            for (int i = 0; i < size; i++) {
                current->data = arr[i];
                current = current->next;
            }

            delete[] arr;
        }
};
