package classifiers.distance_based.elastic_ensemble;

import classifiers.template_classifier.TemplateClassifier;
import distances.dtw.Dtw;
import distances.wdtw.Wdtw;
import evaluation.tuning.ParameterSpace;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class ElasticEnsemble extends TemplateClassifier {

    public static List<Function<Instances, ParameterSpace>> getDefaultDistanceMeasures() {
        return new ArrayList<>(Arrays.asList(
            instances -> Dtw.euclideanParameterSpace(),
            instances -> Dtw.fullWindowParameterSpace(),
            Dtw::discreteParameterSpace,
            instances -> Wdtw.discreteParameterSpace()
            // todo transform for ders

                                            ));
    }

    public ElasticEnsemble() {
        this(getDefaultDistanceMeasures());
    }

    public ElasticEnsemble(Function<Instances, ParameterSpace>... distanceMeasures) {
        this(Arrays.asList(distanceMeasures));
    }

    private final List<Function<Instances, ParameterSpace>> distanceMeasures = new ArrayList<>();

    public ElasticEnsemble(List<Function<Instances, ParameterSpace>> distanceMeasures) {
        setDistanceMeasures(distanceMeasures);
    }


    public void setDistanceMeasures(List<Function<Instances, ParameterSpace>> distanceMeasures) {
        this.distanceMeasures.clear();
        this.distanceMeasures.addAll(distanceMeasures);
    }

    @Override
    public void buildClassifier(final Instances trainInstances) throws
                                                      Exception {

    }
}
