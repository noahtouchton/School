#include <signal.h>
#include <stdlib.h>

int main() {
    int pids[] = {28296, 47830, 47850, 47856, 47888, 47897, 47902, 47907, 47912, 47931};
    for (int i=0; i<10; i++) {
        kill(pids[i], SIGKILL);
    }
    return 0;
}
