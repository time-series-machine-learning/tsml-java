package machine_learning.classifiers.tuned.incremental;

import evaluation.tuning.ParameterSpace;
import utilities.collections.DefaultIterator;
import utilities.params.ParamSet;
import weka.classifiers.Classifier;

import java.util.Iterator;
import java.util.function.Supplier;

public class IncTunedParamSpaceClassifier extends IncTunedClassifier {

    private Iterator<ParamSet> paramSetIterator = new DefaultIterator<ParamSet>() {};
    private Supplier<Classifier> classifierSupplier;
}
