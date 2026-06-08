// include/LinkedList.h
#pragma once
#include <string>
#include <stdexcept>

// 1. Make the Node generic
template <typename T>
struct Node {
    T data;
    Node* next;

    Node(T value) : data(value), next(nullptr) {}
};

// 2. Make the LinkedList generic
template <typename T>
class LinkedList {
    private:
        Node<T>* head;
        int size;

    public:
        LinkedList() : head(nullptr), size(0) {}

        ~LinkedList() {
            while (head) {
                pop();
            }
        }

        void add(T value) {
            Node<T>* newNode = new Node<T>(value);
            if (!head) {
                head = newNode;
            } else {
                Node<T>* current = head;
                while (current->next) {
                    current = current->next;
                }
                current->next = newNode;
            }
            size++;
        }

        void remove(T value) {
            if (!head) return;

            if (head->data == value) {
                Node<T>* temp = head;
                head = head->next;
                delete temp;
                size--;
                return;
            }

            Node<T>* current = head;
            while (current->next && current->next->data != value) {
                current = current->next;
            }

            if (current->next) {
                Node<T>* temp = current->next;
                current->next = current->next->next;
                delete temp;
                size--;
            }
        }

        T pop() {
            if (!head) throw std::runtime_error("List is empty");
            Node<T>* temp = head;
            T value = head->data;
            head = head->next;
            delete temp;
            size--;
            return value;
        }

        bool contains(T value) {
            Node<T>* current = head;
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

            T* arr = new T[size];
            Node<T>* current = head;
            for (int i = 0; i < size; i++) {
                arr[i] = current->data;
                current = current->next;
            }

            for (int i = size - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                std::swap(arr[i], arr[j]);
            }

            current = head;
            for (int i = 0; i < size; i++) {
                current->data = arr[i];
                current = current->next;
            }

            delete[] arr;
        }

        T get(int index) {
            if (index < 0 || index >= size) {
                throw std::out_of_range("Index out of bounds");
            }
            Node<T>* current = head;
            for (int i = 0; i < index; i++) {
                current = current->next;
            }
            return current->data;
        }
};