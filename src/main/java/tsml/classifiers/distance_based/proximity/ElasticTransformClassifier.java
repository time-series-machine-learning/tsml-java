package tsml.classifiers.distance_based.proximity;

import machine_learning.classifiers.ensembles.ContractRotationForest;
import tsml.classifiers.distance_based.utils.classifiers.BaseClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.core.Instance;
import weka.core.Instances;

public class ElasticTransformClassifier extends BaseClassifier {

    private Classifier classifier;
    private ElasticTransform elasticTransform;

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        super.buildClassifier(trainData);
        elasticTransform = new ElasticTransform();
        elasticTransform.setSeed(rand.nextInt());
        Instances transformed = elasticTransform.fitTransform(trainData);
        final ContractRotationForest rotationForest = new ContractRotationForest();
        rotationForest.setMaxNumTrees(1000);
        classifier = rotationForest;
        classifier.buildClassifier(transformed);
    }

    @Override public double[] distributionForInstance(final Instance testCase) throws Exception {
        Instance transformed = elasticTransform.transform(testCase);
        return classifier.distributionForInstance(transformed);
    }
}
