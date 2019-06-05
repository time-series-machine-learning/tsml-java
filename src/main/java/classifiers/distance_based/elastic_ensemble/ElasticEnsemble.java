package classifiers.distance_based.elastic_ensemble;

import classifiers.template_classifier.TemplateClassifier;
import distances.derivative_time_domain.ddtw.CachedDdtw;
import distances.time_domain.dtw.Dtw;
import distances.derivative_time_domain.wddtw.CachedWddtw;
import distances.time_domain.erp.Erp;
import distances.time_domain.lcss.Lcss;
import distances.time_domain.msm.Msm;
import distances.time_domain.twe.Twe;
import distances.time_domain.wdtw.Wdtw;
import evaluation.tuning.ParameterSpace;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class ElasticEnsemble extends TemplateClassifier {

    public static List<Function<Instances, ParameterSpace>> getDefaultParameterSpaceGetters() {
        return new ArrayList<>(Arrays.asList(
            instances -> Dtw.euclideanParameterSpace(),
            instances -> Dtw.fullWindowParameterSpace(),
            Dtw::discreteParameterSpace,
            instances -> CachedDdtw.fullWindowParameterSpace(),
            CachedDdtw::discreteParameterSpace,
            instances -> Wdtw.discreteParameterSpace(),
            instances -> CachedWddtw.discreteParameterSpace(),
            Lcss::discreteParameterSpace,
            Erp::discreteParameterSpace,
            instances -> Msm.discreteParameterSpace(),
            instances -> Twe.discreteParameterSpace()
            ));
    }

    public ElasticEnsemble() {
        this(getDefaultParameterSpaceGetters());
    }

    public ElasticEnsemble(Function<Instances, ParameterSpace>... parameterSpaceGetters) {
        this(Arrays.asList(parameterSpaceGetters));
    }

    private final List<Function<Instances, ParameterSpace>> parameterSpaceGetters = new ArrayList<>();
    private final List<ParameterSpace> parameterSpaces = new ArrayList<>();

    public ElasticEnsemble(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        setParameterSpaceGetters(parameterSpaceGetters);
    }

    public void setParameterSpaceGetters(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        this.parameterSpaceGetters.clear();
        this.parameterSpaceGetters.addAll(parameterSpaceGetters);
    }

    public static List<ParameterSpace> getParameterSpaces(Instances instances, List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        List<ParameterSpace> parameterSpaces = new ArrayList<>();
        for(Function<Instances, ParameterSpace> getter : parameterSpaceGetters) {
            ParameterSpace distanceMeasure = getter.apply(instances);
            parameterSpaces.add(distanceMeasure);
        }
        return parameterSpaces;
    }

    @Override
    public void buildClassifier(final Instances trainInstances) throws
                                                      Exception {
        parameterSpaces.clear();
        parameterSpaces.addAll(getParameterSpaces(trainInstances, parameterSpaceGetters));

    }
}
