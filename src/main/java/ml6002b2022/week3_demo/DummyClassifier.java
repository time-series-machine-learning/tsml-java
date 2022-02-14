package ml6002b2022.week3_demo;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instances;

public class DummyClassifier extends AbstractClassifier {

    @Override
    public Capabilities getCapabilities(){
        return null;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
    }
}
