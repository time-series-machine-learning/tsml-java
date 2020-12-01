package tsml.classifiers;

import weka.classifiers.Classifier;

public class Tmp {
    public static void main(String[] args) {
        final TSClassifier tsc = new TSClassifier() {
            @Override public Classifier getClassifier() {
                return null;
            }
        };
        Classifier classifier = null;
        TSClassifier.wrapClassifier(classifier);
        TSClassifier.wrapClassifier(tsc);
    }
}
