package tsml.classifiers.distance_based.pf;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import tsml.classifiers.distance_based.pf.partition.Partitioner;
import tsml.classifiers.distance_based.pf.partition.ExemplarPartitioner;
import tsml.classifiers.distance_based.pf.relative.ExemplarPicker;
import tsml.classifiers.distance_based.pf.relative.RandomExemplarPicker;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public enum ProximityTreeConfigs { //} implements ClassifierBuilderFactory.ClassifierBuilder {

//    DEFAULT(ProximityTreeConfigs::buildDefaultProximityTree, KnnTag.UNIVARIATE, KnnTag.DISTANCE, KnnTag.SIMILARITY),
//    ;
//
//
//    public String toString() {
//        return classifierBuilder.toString();
//    }
//
//    @Override
//    public String getName() {
//        return classifierBuilder.getName();
//    }
//
//    @Override
//    public Classifier build() {
//        return classifierBuilder.build();
//    }
//
//    @Override
//    public ImmutableList<? extends ClassifierBuilderFactory.Tag> getTags() {
//        return classifierBuilder.getTags();
//    }
//
//    private final ClassifierBuilderFactory.ClassifierBuilder classifierBuilder;
//
//    EeConfig(Supplier<? extends Classifier> supplier, ClassifierBuilderFactory.Tag... tags) {
//        classifierBuilder = new ClassifierBuilderFactory.SuppliedClassifierBuilder(name(), supplier, tags);
//    }
//
//    public static List<ClassifierBuilderFactory.ClassifierBuilder> all() {
//        return Arrays.stream(values()).map(i -> (ClassifierBuilderFactory.ClassifierBuilder) i).collect(Collectors.toList());
//    }

    ;

    public static ProximityTree buildDefaultProximityTree() {
        ProximityTree pt = new ProximityTree();
        pt.setTrainSetupFunction(new Consumer<Instances>() {
            @Override
            public void accept(Instances instances) {
                pt.setPartitionerBuilder(new Function<Instances, Partitioner>() {
                    @Override
                    public Partitioner apply(Instances data) {
                        ExemplarPartitioner split = new ExemplarPartitioner();
                        split.setRandom(pt.getRandom());
                        ExemplarPicker picker = new RandomExemplarPicker();
                        picker.setRandom(pt.getRandom());
                        List<Instance> exemplars = picker.pickExemplars(data);
                        split.setExemplars(exemplars);
                        // todo setup dist measures
                        return split;
                    }
                });
            }
        });
        return pt;
    }

}
