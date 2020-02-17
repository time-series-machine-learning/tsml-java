package utilities;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import weka.core.Randomizable;

import java.util.Random;

public interface Rand extends Randomizable {
    void setRandom(Random random);
    Random getRandom();
}
