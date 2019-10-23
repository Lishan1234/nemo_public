
long fake_main(const char * name, const char * b, const char * d, const char * i, const char * o, int minutes, int fps);

int test(const char * name, const char * b, const char * d, const char * i, const char * o);

bool within_one_second(struct timeval * before, struct timeval * after);
