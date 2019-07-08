package utilities.samplers;

import weka.core.Instances;

public interface Sampler {
    void setInstances(Instances instances);

    boolean hasNext();

    Object next();
}
