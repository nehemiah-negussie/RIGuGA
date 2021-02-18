#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <typeinfo>

using namespace cv;
using namespace std;

int rows;
int cols;

class Member {
    public:
        vector<double> dna;
        double fitness;
        bool operator< (const Member &other) const {
            return fitness > other.fitness;
        }
};

// number 0 to 1 exclusive
double randomNumber (){
    mt19937_64 rng;
    uint64_t timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
    seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    uniform_real_distribution<double> unif(0,1);
    double currentRandomNumber = unif(rng);
    return currentRandomNumber;
}

Mat shapesToImage(vector<double> dna){
    Mat canvas = Mat::zeros(rows, cols, CV_8UC3);
    Mat shapecanvas;
    for (int i = 0; i < 50*8; i+= 8){
        double x1 = ceil(dna[i] * cols);
        double y1 = ceil(dna[i+1] * rows);
        double x2 = ceil(dna[i+2] * cols);
        double y2 = ceil(dna[i+3] * rows);
        double r = floor(dna[i+4] * 256);
        double g = floor(dna[i+5] * 256);
        double b = floor(dna[i+6] * 256);
        double a = dna[i+7]; 
        canvas.copyTo(shapecanvas);
        rectangle(shapecanvas, Point(x1, y1), Point(x2, y2), Scalar(b,g,r), 
            FILLED, LINE_4);
        addWeighted (shapecanvas, a, canvas, 1.0 - a, 0.0, canvas);
    }
    return canvas;
}

double fitness(Member person){
    Mat img = imread("/Users/nemo/Downloads/george.jpg", IMREAD_COLOR);
    Mat canvas = shapesToImage(person.dna);
    Mat canvasHSV;
    Mat imgHSV;
    // resize images to 10% of size
    resize(img, imgHSV, Size(75, 75));
    resize(canvas, canvasHSV, Size(75, 75));
    int workingRows = imgHSV.rows;
    int workingCols = imgHSV.cols;
    // convert 3x8bit image into hsv colorspace (hue will be 180 not 360)
    cvtColor(canvasHSV, canvasHSV, COLOR_BGR2HSV);
    cvtColor(imgHSV, imgHSV, COLOR_BGR2HSV);
    double difference = 0;
    for (int i = 0; i < workingCols; i++){
        for (int j = 0; j < workingRows; j++){
            Point pt(i, j);
            Vec3b hsv1 = canvasHSV.at<Vec3b>(pt);
            Vec3b hsv2 = imgHSV.at<Vec3b>(pt);
            double pixelDiff = abs(hsv1(0) - hsv2(0));
            double corrected = min(pixelDiff, 180-pixelDiff);
            difference += corrected;
        }
    }
    double fitness = difference/(workingCols*workingRows*180);
    // cout << fitness << endl;
    return fitness;
}

vector<Member> initialization(int n){
    vector<Member> population;
    for (int i = 0; i < n; i++){
        Member person;
        for (int j = 0; j < 50; j++){
            // push x1, y1, x2, y2, r, g, b, a
            for (int k = 0; k < 8; k++){
                person.dna.push_back(randomNumber());
            }
        }
        person.fitness = fitness(person);
        population.push_back(person);
    }
    cout << "Initialization finished!" << endl;
    return population;
}

vector<Member> fitnessSort (vector<Member> population){
    vector<Member> pop_copy = population;
    sort(pop_copy.begin(), pop_copy.end());
    return pop_copy;
}



Member breed(Member mom, Member dad){
    vector<double> breedDNA;
    Member child;
    for (int i = 0; i < 50 * 8; i += 8){
        if (randomNumber() < 0.5) 
            breedDNA = mom.dna;
        else 
            breedDNA = dad.dna;
        for (int j = 0; j < 8; j++){
            if (randomNumber() < 0.01){
                breedDNA[i+j] += randomNumber() * 0.1 * 2 - 0.1;
                if (breedDNA[i+j] < 0) breedDNA[i+j] = 0;
                if (breedDNA[i+j] > 1) breedDNA[i+j] = 1;
            }
            child.dna.push_back(breedDNA[i+j]);
        }
    }
    child.fitness = fitness(child);
    return child;
}

int main() {
    Mat img = imread("/Users/nemo/Downloads/george.jpg", IMREAD_COLOR);
    rows = img.rows;
    cols = img.cols;
    // takes O(n)
    vector<Member> population = initialization(200);
    vector<Member> children;
    vector<Member> parentPool;

    for (int generations = 0; generations < 14000; generations++){
        double temp = (generations+1)/4000;
        cout << "Generation: " << generations << endl;
        // n log n
        population = fitnessSort(population);
        cout << "Highest fitness: " << population[0].fitness << endl;
        destroyAllWindows();
        // imshow ("Gen", shapesToImage(population[0].dna));
        waitKey(1); 
        for (int i = 0; i < max(15, (int) ceil(population.size() * min(0.2, (0.2*temp)))); i++){
            parentPool.push_back(population[i]);
        }
        
        for (int i = 0; i < parentPool.size(); i++){
            children.push_back(breed(parentPool[i], parentPool[rand() % parentPool.size()]));
        }
        children = fitnessSort(children);
        population.resize(population.size()-children.size());
        population.insert(population.end(), children.begin(), children.end());
        parentPool.clear();
        children.clear();
    }
    return 0;
}