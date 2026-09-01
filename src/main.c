#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <pthread.h>
#include <unistd.h>
#include <libgen.h>
#include <signal.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define CACHE_FILE "photag.cache"
#define NK_IMPLEMENTATION
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_GLFW_GL3_IMPLEMENTATION
#include "../include/nuklear.h"
#include "../include/nuklear_glfw_gl3.h"
#include "../include/darknet.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024
#define THRESHOLD 0.5f
#define NMS_THRESHOLD 0.45f
#define MAX_IMAGES 1000
#define IMAGES_PER_PAGE 20

network *net = NULL;
char **class_names = NULL;
int class_count = 0;
volatile int scan_progress = 0;
volatile int scan_total = 0;
volatile int is_scanning = 0;

struct ScanJob {
    struct AppImage* images;
    int image_count;
};

// Helper struct to hold our loaded texture data
struct AppImage {
    char filename[256];
    char filepath[512];
    struct nk_image nk_img;
    char tags[512];
    int is_scanned;
    GLuint gl_tex;
    int is_loaded;
};

// Crash handler callback
void crash_handler(int sig) {
    const char* sig_name = "UNKNOWN";
    switch(sig) {
        case SIGSEGV: sig_name = "SIGSEGV (Segmentation Fault)"; break;
        case SIGABRT: sig_name = "SIGABRT (Abort Signal)"; break;
        case SIGFPE:  sig_name = "SIGFPE (Arithmetic Exception)"; break;
        case SIGILL:  sig_name = "SIGILL (Illegal Instruction)"; break;
    }

    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, " CRASH DETECTED: %s\n", sig_name);
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "Press ENTER to close the application...\n");
    fflush(stderr);

    getchar();
    exit(sig);
}

// Function to load an image file into an OpenGL texture
struct nk_image load_texture(const char* filepath, GLuint* out_tex) {
    int x, y, n;
    unsigned char *data = stbi_load(filepath, &x, &y, &n, 4);
    if (!data) {
        printf("Failed to load: %s\n", filepath);
        return nk_image_id(0);
    }

    glGenTextures(1, out_tex);
    glBindTexture(GL_TEXTURE_2D, *out_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    stbi_image_free(data);
    return nk_image_id((int)*out_tex);
}

// Function to manage VRAM: unloads old page, loads new page
void load_page_textures(struct AppImage* images, int total_images, int target_page) {
    int start_idx = target_page * IMAGES_PER_PAGE;
    int end_idx = start_idx + IMAGES_PER_PAGE;

    // 1. Unload textures that are outside the current page
    for (int i = 0; i < total_images; i++) {
        if (images[i].is_loaded && (i < start_idx || i >= end_idx)) {
            glDeleteTextures(1, &images[i].gl_tex);
            images[i].nk_img = nk_image_id(0);
            images[i].is_loaded = 0;
        }
    }

    // 2. Load textures for the new page
    for (int i = start_idx; i < end_idx && i < total_images; i++) {
        if (!images[i].is_loaded) {
            images[i].nk_img = load_texture(images[i].filepath, &images[i].gl_tex);
            images[i].is_loaded = 1;
        }
    }
}

enum AppState { STATE_GALLERY, STATE_VIEWER };

// Helper function to load class names from a file
char** load_class_names(const char *filename, int *num_classes) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Failed to open class names file: %s\n", filename);
        return NULL;
    }

    // Count lines first
    int count = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        count++;
    }
    rewind(fp);

    // Allocate memory for class names
    char **names = (char**)malloc(count * sizeof(char*));
    if (!names) {
        fclose(fp);
        return NULL;
    }

    // Read class names
    int i = 0;
    while (fgets(line, sizeof(line), fp) && i < count) {
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        names[i] = (char*)malloc(strlen(line) + 1);
        if (names[i]) {
            strcpy(names[i], line);
        }
        i++;
    }
    fclose(fp);
    *num_classes = count;
    return names;
}

// Function to free class names
void free_class_names(char **names, int count) {
    if (!names) return;
    for (int i = 0; i < count; i++) {
        if (names[i]) free(names[i]);
    }
    free(names);
}

void run_darknet_scan(struct AppImage* img) {
    if (!net || !class_names) {
        printf("Error: Darknet not initialized!\n");
        return;
    }

    printf("Scanning: %s\n", img->filename);

    image im = load_image_color(img->filepath, 0, 0);
    image sized = letterbox_image(im, net->w, net->h);

    layer l = net->layers[net->n - 1];
    network_predict_ptr(net, sized.data);

    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, THRESHOLD, 0.2f, NULL, 1, &nboxes, 1);

    if (NMS_THRESHOLD > 0) {
        do_nms_sort(dets, nboxes, l.classes, NMS_THRESHOLD);
    }

    img->tags[0] = '\0';
    int tags_added = 0;

    for (int i = 0; i < nboxes; ++i) {
        for (int j = 0; j < l.classes; ++j) {
            if (dets[i].prob[j] > THRESHOLD) {
                if (!strstr(img->tags, class_names[j])) {
                    if (tags_added > 0) {
                        strncat(img->tags, ", ", sizeof(img->tags) - strlen(img->tags) - 1);
                    }
                    strncat(img->tags, class_names[j], sizeof(img->tags) - strlen(img->tags) - 1);
                    tags_added++;
                }
            }
        }
    }

    if (tags_added == 0) {
        strcpy(img->tags, "unclassified");
    }

    img->is_scanned = 1;

    free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);
}

void update_vram(struct AppImage* all_images, int total_count, struct AppImage** visible_images, int visible_count, int target_page) {
    int start_idx = target_page * IMAGES_PER_PAGE;
    int end_idx = start_idx + IMAGES_PER_PAGE;

    for (int i = 0; i < total_count; i++) {
        int should_be_loaded = 0;

        for (int v = start_idx; v < end_idx && v < visible_count; v++) {
            if (&all_images[i] == visible_images[v]) {
                should_be_loaded = 1;
                break;
            }
        }

        if (all_images[i].is_loaded && !should_be_loaded) {
            glDeleteTextures(1, &all_images[i].gl_tex);
            all_images[i].is_loaded = 0;
        }
        else if (!all_images[i].is_loaded && should_be_loaded) {
            all_images[i].nk_img = load_texture(all_images[i].filepath, &all_images[i].gl_tex);
            all_images[i].is_loaded = 1;
        }
    }
}

char* strcasestr_custom(const char* haystack, const char* needle) {
    if (!*needle) return (char*)haystack;
    for (; *haystack; ++haystack) {
        if (tolower((unsigned char)*haystack) == tolower((unsigned char)*needle)) {
            const char *h = haystack, *n = needle;
            while (*h && *n && tolower((unsigned char)*h) == tolower((unsigned char)*n)) {
                ++h; ++n;
            }
            if (!*n) return (char*)haystack;
        }
    }
    return NULL;
}

void load_cache(struct AppImage* images, int image_count) {
    FILE* f = fopen(CACHE_FILE, "r");
    if (!f) return;

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\r\n")] = 0;

        char* delim = strchr(line, '|');
        if (delim) {
            *delim = '\0';
            char* cached_filename = line;
            char* cached_tags = delim + 1;

            for (int i = 0; i < image_count; i++) {
                if (strcmp(images[i].filename, cached_filename) == 0) {
                    strncpy(images[i].tags, cached_tags, sizeof(images[i].tags) - 1);
                    images[i].is_scanned = 1;
                    break;
                }
            }
        }
    }
    fclose(f);
    printf("Loaded tags from cache.\n");
}

void save_cache(struct AppImage* images, int image_count) {
    FILE* f = fopen(CACHE_FILE, "w");
    if (!f) {
        printf("Error: Could not open cache file for writing.\n");
        return;
    }

    for (int i = 0; i < image_count; i++) {
        if (images[i].is_scanned) {
            fprintf(f, "%s|%s\n", images[i].filename, images[i].tags);
        }
    }
    fclose(f);
    printf("Cache saved successfully.\n");
}

void init_darknet() {
    const char *home = getenv("HOME");

    char cfg_file[512], weights_file[512], names_file[512];
    snprintf(cfg_file, sizeof(cfg_file), "%s/clones/photag/yolov3.cfg", home);
    snprintf(weights_file, sizeof(weights_file), "%s/clones/photag/yolov3.weights", home);
    snprintf(names_file, sizeof(names_file), "%s/clones/photag/coco.names", home);

    printf("Loading Darknet model from %s... This might take a second.\n", home);

    net = load_network(cfg_file, weights_file, 0);
    set_batch_network(net, 1);

    class_names = load_class_names(names_file, &class_count);
    if (!class_names) {
        printf("Warning: Failed to load class names. Defaulting to %d classes.\n", class_count);
        class_count = 80;
    }

    printf("Darknet initialized successfully!\n");
}

void* background_scan_thread(void* arg) {
    struct ScanJob* job = (struct ScanJob*)arg;

    init_darknet();

    for (int i = 0; i < job->image_count; i++) {
        if (!job->images[i].is_scanned) {
            run_darknet_scan(&job->images[i]);
            scan_progress++;
        }
    }

    save_cache(job->images, job->image_count);

    if (net) {
        free_network_ptr(net);
        net = NULL;
    }
    if (class_names) {
        free_class_names(class_names, class_count);
        class_names = NULL;
    }

    is_scanning = 0;
    free(job);
    return NULL;
}

int main(void) {
    // 0. Register Crash Signal Handlers
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGFPE,  crash_handler);
    signal(SIGILL,  crash_handler);

    // 1. Initialize GLFW and OpenGL Context
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *win = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Photag", NULL, NULL);
    glfwMakeContextCurrent(win);
    glewInit();

    // 2. Initialize Nuklear
    static struct nk_glfw glfw = {0};
    struct nk_context *ctx = nk_glfw3_init(&glfw, win, NK_GLFW3_INSTALL_CALLBACKS);
    struct nk_font_atlas *atlas;
    nk_glfw3_font_stash_begin(&glfw, &atlas);
    nk_glfw3_font_stash_end(&glfw);

    // 3. Scan the directory
    struct AppImage images[MAX_IMAGES];
    memset(images, 0, sizeof(images));
    int image_count = 0;
    const char* target_dir = ".";

    DIR *dir = opendir(target_dir);
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL && image_count < MAX_IMAGES) {
            if (strstr(entry->d_name, ".jpg") || strstr(entry->d_name, ".png")) {
                snprintf(images[image_count].filepath, sizeof(images[0].filepath), "%s/%s", target_dir, entry->d_name);
                strncpy(images[image_count].filename, entry->d_name, 255);
                images[image_count].is_loaded = 0;
                image_count++;
            }
        }
        closedir(dir);
    } else {
        printf("Could not open directory %s. Please create it and add some images.\n", target_dir);
    }
    load_cache(images, image_count);

    // 4. Pagination Setup
    int current_page = 0;

    enum AppState current_state = STATE_GALLERY;
    int selected_image_idx = -1;
    char search_query[256] = {0};
    struct AppImage* filtered_images[MAX_IMAGES];
    int filtered_count = 0;

    if (image_count > 0) {
        for(int i=0; i<image_count; i++) filtered_images[i] = &images[i];
        update_vram(images, image_count, filtered_images, image_count, 0);
    }

    // 5. Main GUI Loop
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        nk_glfw3_new_frame(&glfw);

        int w, h;
        glfwGetWindowSize(win, &w, &h);

        // --- FILTER LOGIC ---
        filtered_count = 0;
        for (int i = 0; i < image_count; i++) {
            if (search_query[0] == '\0' ||
                strcasestr_custom(images[i].filename, search_query) ||
                (images[i].is_scanned && strcasestr_custom(images[i].tags, search_query))) {

                filtered_images[filtered_count++] = &images[i];
            }
        }

        int total_pages = (filtered_count + IMAGES_PER_PAGE - 1) / IMAGES_PER_PAGE;
        if (current_page >= total_pages && total_pages > 0) {
            current_page = total_pages - 1;
            update_vram(images, image_count, filtered_images, filtered_count, current_page);
        }

        if (nk_begin(ctx, "PhoTag", nk_rect(0, 0, w, h), NK_WINDOW_BACKGROUND | NK_WINDOW_NO_SCROLLBAR)) {

            // --- TOP BAR: SEARCH & SCAN ---
            nk_layout_row_dynamic(ctx, 35, 2);

            nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, search_query, sizeof(search_query) - 1, nk_filter_default);

            if (!is_scanning) {
                if (nk_button_label(ctx, "Scan Uncached Images")) {
                    scan_total = 0;
                    scan_progress = 0;
                    for(int i = 0; i < image_count; i++) {
                        if(!images[i].is_scanned) scan_total++;
                    }

                    if (scan_total > 0) {
                        is_scanning = 1;
                        struct ScanJob* job = malloc(sizeof(struct ScanJob));
                        job->images = images;
                        job->image_count = image_count;

                        pthread_t thread_id;
                        pthread_create(&thread_id, NULL, background_scan_thread, job);
                        pthread_detach(thread_id);
                    }
                }
            } else {
                nk_size prog = (nk_size)scan_progress;
                nk_progress(ctx, &prog, scan_total, NK_FIXED);
            }

            if (current_state == STATE_GALLERY) {
                // --- GALLERY CONTROLS ---
                nk_layout_row_dynamic(ctx, 30, 3);
                if (nk_button_label(ctx, "<< Prev") && current_page > 0) {
                    current_page--;
                    update_vram(images, image_count, filtered_images, filtered_count, current_page);
                }

                char pg[64];
                snprintf(pg, sizeof(pg), "Page %d of %d", current_page + 1, total_pages == 0 ? 1 : total_pages);
                nk_label(ctx, pg, NK_TEXT_CENTERED);

                if (nk_button_label(ctx, "Next >>") && current_page < total_pages - 1) {
                    current_page++;
                    update_vram(images, image_count, filtered_images, filtered_count, current_page);
                }

                // --- GALLERY GRID ---
                int start = current_page * IMAGES_PER_PAGE;
                nk_layout_row_dynamic(ctx, 180, (w / 160) > 0 ? (w / 160) : 1);

                for (int i = start; i < start + IMAGES_PER_PAGE && i < filtered_count; i++) {
                    struct AppImage* img = filtered_images[i];

                    if (nk_group_begin(ctx, img->filename, NK_WINDOW_BORDER | NK_WINDOW_NO_SCROLLBAR)) {
                        nk_layout_row_dynamic(ctx, 110, 1);

                        if (img->is_loaded) {
                            if (nk_button_image(ctx, img->nk_img)) {
                                selected_image_idx = (img - images);
                                current_state = STATE_VIEWER;
                            }
                        } else {
                            nk_label(ctx, "...", NK_TEXT_CENTERED);
                        }

                        nk_layout_row_dynamic(ctx, 15, 1);
                        nk_label(ctx, img->filename, NK_TEXT_CENTERED);

                        if (img->is_scanned) {
                            nk_label(ctx, img->tags, NK_TEXT_CENTERED);
                        } else {
                            nk_label(ctx, "(Unscanned)", NK_TEXT_CENTERED);
                        }

                        nk_group_end(ctx);
                    }
                }
            }
            else if (current_state == STATE_VIEWER) {
                nk_layout_row_dynamic(ctx, 30, 2);
                if (nk_button_label(ctx, "Back to Gallery")) {
                    current_state = STATE_GALLERY;
                }
                nk_label(ctx, images[selected_image_idx].tags, NK_TEXT_LEFT);

                nk_layout_row_dynamic(ctx, h - 100, 1);
                nk_image(ctx, images[selected_image_idx].nk_img);
            }
        }
        nk_end(ctx);

        glViewport(0, 0, w, h);
        glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        nk_glfw3_render(&glfw, NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
        glfwSwapBuffers(win);
    }

    // 6. Cleanup
    for (int i = 0; i < image_count; i++) {
        if (images[i].is_loaded) {
            glDeleteTextures(1, &images[i].gl_tex);
        }
    }
    nk_glfw3_shutdown(&glfw);
    glfwTerminate();
    return 0;
}