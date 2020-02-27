package utilities;

import weka.core.Randomizable;

import java.util.Random;

public interface Randomised extends Randomizable {

    Random getRandom();

    void setRandom(Random random);
}
