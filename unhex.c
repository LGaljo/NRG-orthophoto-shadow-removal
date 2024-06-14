#include<stdio.h>
#include<unistd.h>

int main(){
    char offs[16];
    char line[16];
    int c, i;
    while(scanf("%s", offs) == 1){
        /* fprintf(stderr, "Hhhhheeeee\n"); */
        for (i = 0; i < 16; i++){
            scanf("%hhx", &line[i]);
        }
        scanf(" s %*s");
        fwrite(line, 16, 1, stdout);
    }
}
