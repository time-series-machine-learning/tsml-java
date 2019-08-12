package timeseriesweka.classifiers.distance_based.ee;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class TunedClassifier
    extends AbstractClassifier {
    private final IncrementalTuner tuner;

    public TunedClassifier(final IncrementalTuner tuner) {this.tuner = tuner;}


    @Override
    public void buildClassifier(final Instances data) throws
                                                      Exception {
        tuner.setInstances(data);
        while (tuner.hasNext()) {
            tuner.next();
        }
    }
}
