package tsml.classifiers.distance_based.proximity;

import java.util.ArrayList;
import java.util.List;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import tsml.filters.Utilities;
import utilities.ArrayUtilities;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ProxForest extends BaseClassifier {

    public interface ConstituentBuilder {
        Classifier build(Instances data);
    }

    private List<Classifier> constituents = new ArrayList<>();
    private ConstituentBuilder constituentBuilder = (Instances trainData) -> {
        throw new UnsupportedOperationException();
    };
    private int numTrees = 100;

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        final boolean rebuild = isRebuild();
        super.buildClassifier(trainData);
        if(rebuild) {
            constituents = new ArrayList<>();
        }
        while(constituents.size() < numTrees) { // todo different stopping condition
            Classifier constituent = constituentBuilder.build(trainData);
            constituents.add(constituent);
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distribution = new double[getNumClasses()];
        for(Classifier constituent : constituents) {
            // todo fix python pf to match
            // todo use different voting methods / weighting methods
//            double[] constituentDistribution = constituent.distributionForInstance(instance);
            double classLabel = constituent.classifyInstance(instance);
            distribution[(int) classLabel]++;
        }
        ArrayUtilities.normalise(distribution);
        return distribution;
    }
}
