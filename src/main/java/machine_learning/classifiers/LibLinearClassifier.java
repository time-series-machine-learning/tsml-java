package machine_learning.classifiers;

import de.bwaldvogel.liblinear.*;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class LibLinearClassifier extends AbstractClassifier {

    double bias = 1;
    public SolverType solverType = SolverType.L2R_L2LOSS_SVC;
    int iterations = 5000;
    double e = 0.1;
    double p = 0.1;
    double c = 1;

    Model linearModel;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        FeatureNode[][] features = new FeatureNode[data.numInstances()][];
        double[] labels = new double[features.length];
        for (int i = 0; i < features.length; i++){
            Instance inst = data.get(i);
            features[i] = new FeatureNode[data.numAttributes()-1];

            for (int n = 0; n < features[i].length; n++){
                features[i][n] = new FeatureNode(n + 1, inst.value(n));
            }

            labels[i] = inst.classValue();
        }

        Problem problem = new Problem();
        problem.bias = bias;
        problem.y = labels;
        problem.n = data.numAttributes();
        problem.l = features.length;
        problem.x = features;

        Parameter par = new Parameter(solverType, c, e, iterations, p);
        linearModel = Linear.train(problem, par);
    }

    @Override
    public double classifyInstance(Instance inst) throws Exception {
        FeatureNode[] features = new FeatureNode[inst.numAttributes()-1];
        for (int n = 0; n < features.length; n++){
            features[n] = new FeatureNode(n + 1, inst.value(n));
        }
        return Linear.predict(linearModel, features);
    }

    public double[] distributionForInstance(Instance inst) throws Exception {
        FeatureNode[] features = new FeatureNode[inst.numAttributes()-1];
        for (int n = 0; n < features.length; n++){
            features[n] = new FeatureNode(n + 1, inst.value(n));
        }
        double[] probs = new double[inst.dataset().numClasses()];
        Linear.predictProbability(linearModel, features, probs);
        return probs;
    }
}
