package tsml.classifiers.distance_based.utils.classifier_building;

import com.google.common.collect.ImmutableSet;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.elastic_ensemble.ElasticEnsemble;
import tsml.classifiers.distance_based.knn.KNNLOOCV;
import weka.classifiers.Classifier;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class ClassifierBuilderFactory {

    public interface ClassifierBuilder extends Supplier<Classifier> {
        String toString();
        String getName();
        Classifier build();

        @Override
        default Classifier get() {
            return build();
        }
    }

    public static class SuppliedClassifierBuilder implements ClassifierBuilder {
        private final String name;
        private final Supplier<? extends Classifier> supplier;

        public SuppliedClassifierBuilder(final String name, final Supplier<? extends Classifier> supplier) {
            this.name = name;
            this.supplier = supplier;
        }

        public String getName() {
            return name;
        }

        public Supplier<? extends Classifier> getSupplier() {
            return supplier;
        }

        public Classifier build() {
            Classifier classifier = getSupplier().get();
            if(classifier instanceof EnhancedAbstractClassifier) {
                ((EnhancedAbstractClassifier) classifier).setClassifierName(getName());
            }
            return classifier;
        }

        @Override public String toString() {
            return name;
        }
    }

    private static ClassifierBuilderFactory INSTANCE;
    private final Map<String, ClassifierBuilder> classifierBuildersByName = new HashMap<>();
    private final Set<ClassifierBuilder> classifierBuilders = new HashSet<>();

    public ClassifierBuilderFactory() {}

    @Override public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(getClass().getSimpleName()).append("{").append(System.lineSeparator());
        for(Map.Entry<String, ClassifierBuilder> entry : classifierBuildersByName.entrySet()) {
            stringBuilder.append("\t");
            stringBuilder.append(entry.getKey());
            stringBuilder.append(": ");
            stringBuilder.append(System.lineSeparator());
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }

    public static ClassifierBuilderFactory getGlobalInstance() {
        if(INSTANCE == null) {
            INSTANCE = new ClassifierBuilderFactory();
            INSTANCE.addAll(KNNLOOCV.FACTORY);
            INSTANCE.addAll(ElasticEnsemble.FACTORY);
        }
        return INSTANCE;
    }

    public Set<ClassifierBuilder> all() {
        return new HashSet<>(classifierBuilders);
    }

    public void addAll(ClassifierBuilderFactory other) {
        addAll(other.classifierBuilders);
    }

    public void addAll(Collection<ClassifierBuilder> all) {
        for(ClassifierBuilder builder : all) {
            add(builder);
        }
    }

    public void addAll(ClassifierBuilder... builders) {
        addAll(Arrays.asList(builders));
    }

    public void addAll(Supplier<ClassifierBuilder>... suppliers) {
        addAll(Arrays.stream(suppliers).map(Supplier::get).collect(Collectors.toList()));
    }

    public ClassifierBuilder add(ClassifierBuilder classifierBuilder) {
        String name = classifierBuilder.getName();
        name = name.toLowerCase();
        if(classifierBuildersByName.containsKey(name)) {
            throw new IllegalArgumentException("oops, a classifier already exists under the name: " + name);
        } else if(classifierBuilders.contains(classifierBuilder)) {
            throw new IllegalArgumentException("oops, a classifier already exists under that supplier.");
        } else {
            classifierBuilders.add(classifierBuilder);
            classifierBuildersByName.put(name, classifierBuilder);
        }
        return classifierBuilder;
    }

    public ClassifierBuilder getClassifierBuilderByName(String name) {
        name = name.toLowerCase();
        ClassifierBuilder classifierBuilder = classifierBuildersByName.get(name);
        return classifierBuilder;
    }

    public Set<ClassifierBuilder> getClassifierBuildersByNames(String... names) {
        return getClassifierBuildersByNames(Arrays.asList(names));
    }

    public Set<ClassifierBuilder> getClassifierBuildersByNames(Iterable<String> names) {
        Set<ClassifierBuilder> set = new HashSet<>();
        for(String name : names) {
            set.add(getClassifierBuilderByName(name));
        }
        return ImmutableSet.copyOf(set);
    }

    public static void main(String[] args) {
        System.out.println(getGlobalInstance().toString());
    }
}
