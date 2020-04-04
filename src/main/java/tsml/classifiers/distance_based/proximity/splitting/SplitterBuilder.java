package tsml.classifiers.distance_based.proximity.splitting;

import java.io.Serializable;
import java.util.Random;
import weka.core.Instances;

public interface SplitterBuilder extends Serializable {

    SplitterBuilder setRandom(Random randomSource);

    SplitterBuilder setData(Instances data);

    Splitter build();

    Instances getData();

    Random getRandom();
}
