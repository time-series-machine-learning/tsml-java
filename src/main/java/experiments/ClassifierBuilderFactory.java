package experiments;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.ee.CEE;
import tsml.classifiers.distance_based.knn.configs.KnnConfigs;
import utilities.Utilities;
import weka.classifiers.Classifier;

import java.util.*;
import java.util.function.Supplier;

public class ClassifierBuilderFactory {

    public static class ClassifierBuilder<A extends Classifier> {
        private final String name;
        private final ImmutableList<String> tags;
        private final Supplier<A> supplier;

        public ClassifierBuilder(final String name,
                                 final Supplier<A> supplier,
                                 final String... tags) {
            this(name, supplier, Arrays.asList(tags));
        }

        public ClassifierBuilder(final String name, final Supplier<A> supplier, final Iterable<String> tags) {
            this.name = name;
            List<String> lowerCaseTags = Utilities.convert(tags, String::toLowerCase);
            this.tags = ImmutableList.copyOf(lowerCaseTags);
            this.supplier = supplier;
        }

        public String getName() {
            return name;
        }

        public ImmutableList<String> getTags() {
            return tags;
        }

        public Supplier<A> getSupplier() {
            return supplier;
        }

        public A build() {
            A classifier = getSupplier().get();
            if(classifier instanceof EnhancedAbstractClassifier) {
                ((EnhancedAbstractClassifier) classifier).setClassifierName(getName());
            }
            // todo we could also set tags then the classifier knows it's capabilities...
            return classifier;
        }

        @Override public String toString() {
            return name;
        }
    }

    private static ClassifierBuilderFactory INSTANCE = new ClassifierBuilderFactory();
    private final Map<String, ClassifierBuilder<?>> classifierBuildersByName = new TreeMap<>();
    private final Map<String, Set<ClassifierBuilder<?>>> classifierBuildersByTag = new TreeMap<>();
    private final Map<Supplier<?>, ClassifierBuilder<?>> classifierBuildersBySupplier = new HashMap<>();

    public ClassifierBuilderFactory() {}

    @Override public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(getClass().getSimpleName()).append("{").append(System.lineSeparator());
        for(Map.Entry<String, ClassifierBuilder<?>> entry : classifierBuildersByName.entrySet()) {
            stringBuilder.append("\t");
            stringBuilder.append(entry.getKey());
            stringBuilder.append(": ");
            stringBuilder.append(entry.getValue().getTags().toString());
            stringBuilder.append(System.lineSeparator());
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }

    public static ClassifierBuilderFactory getInstance() {
        return INSTANCE;
    }

    public void add(ClassifierBuilder<?> classifierBuilder) {
        String name = classifierBuilder.getName();
        name = name.toLowerCase();
        Supplier<?> supplier = classifierBuilder.getSupplier();
        if(classifierBuildersByName.containsKey(name)) {
            throw new IllegalArgumentException("oops, a classifier already exists under the name: " + name);
        } else if(classifierBuildersBySupplier.containsKey(supplier)) {
            throw new IllegalArgumentException("oops, a classifier already exists under that supplier.");
        } else {
            classifierBuildersBySupplier.put(supplier, classifierBuilder);
            classifierBuildersByName.put(name, classifierBuilder);
        }
        for(String tag : classifierBuilder.getTags()) {
            classifierBuildersByTag.computeIfAbsent(tag, k -> new HashSet<>()).add(classifierBuilder);
        }
    }

    public static void addGlobal(ClassifierBuilder<?> classifierBuilder) {
        getInstance().add(classifierBuilder);
    }

    public ClassifierBuilder<?> getClassifierBuilderByName(String name) {
        name = name.toLowerCase();
        ClassifierBuilder<?> classifierBuilder = classifierBuildersByName.get(name);
        if(classifierBuilder == null) {
            throw new IllegalArgumentException("oops, there's no classifier by the name: " + name);
        }
        return classifierBuilder;
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByNames(String... names) {
        return getClassifierBuildersByNames(Arrays.asList(names));
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByNames(Iterable<String> names) {
        Set<ClassifierBuilder<?>> set = new HashSet<>();
        for(String name : names) {
            set.add(getClassifierBuilderByName(name));
        }
        return ImmutableSet.copyOf(set);
    }

    public static ClassifierBuilder<?> getClassifierBuilderByNameGlobal(String name) {
        return getInstance().getClassifierBuilderByName(name);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByNamesGlobal(String... names) {
        return getInstance().getClassifierBuildersByNames(names);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByNamesGlobal(Iterable<String> names) {
        return getInstance().getClassifierBuildersByNames(names);
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByTag(String tag) {
        tag = tag.toLowerCase();
        Set<ClassifierBuilder<?>> classifierBuilders = classifierBuildersByTag.get(tag);
        if(classifierBuilders != null) {
            return ImmutableSet.copyOf(classifierBuilders);
        } else {
            return ImmutableSet.of();
        }
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByTags(Iterable<String> tags) {
        Set<ClassifierBuilder<?>> set = new HashSet<>();
        for(String tag : tags) {
            set.addAll(getClassifierBuildersByTag(tag));
        }
        return ImmutableSet.copyOf(set);
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByTags(String... tags) {
        return getClassifierBuildersByTags(Arrays.asList(tags));
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByTagGlobal(String tag) {
        return getInstance().getClassifierBuildersByTag(tag);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByTagsGlobal(Iterable<String> tags) {
        return getInstance().getClassifierBuildersByTags(tags);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByTagsGlobal(String... tags) {
        return getInstance().getClassifierBuildersByTags(tags);
    }

    static {
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("ed-1nn-v1", KnnConfigs::buildEd1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("ed-1nn-v2", KnnConfigs::buildEd1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("dtw-1nn-v1", KnnConfigs::buildDtw1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("dtw-1nn-v2", KnnConfigs::buildDtw1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("ddtw-1nn-v1", KnnConfigs::buildDdtw1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("ddtw-1nn-v2", KnnConfigs::buildDdtw1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-dtw-1nn-v1", KnnConfigs::buildTunedDtw1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-dtw-1nn-v2", KnnConfigs::buildTunedDtw1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-ddtw-1nn-v1",
                                                                    KnnConfigs::buildTunedDdtw1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-ddtw-1nn-v2",
                                                                    KnnConfigs::buildTunedDdtw1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-wdtw-1nn-v1",
                                                                    KnnConfigs::buildTunedWdtw1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-wdtw-1nn-v2",
                                                                    KnnConfigs::buildTunedWdtw1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-wddtw-1nn-v1",
                                                                    KnnConfigs::buildTunedWddtw1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-wddtw-1nn-v2",
                                                                    KnnConfigs::buildTunedWddtw1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-msm-1nn-v1", KnnConfigs::buildTunedMsm1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-msm-1nn-v2", KnnConfigs::buildTunedMsm1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-lcss-1nn-v1",
                                                                    KnnConfigs::buildTunedLcss1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-lcss-1nn-v2",
                                                                    KnnConfigs::buildTunedLcss1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-erp-1nn-v1",
                                                                    KnnConfigs::buildTunedErp1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-erp-1nn-v2",
                                                                    KnnConfigs::buildTunedErp1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-twed-1nn-v1",
                                                                    KnnConfigs::buildTunedTwed1nnV1,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("tuned-twed-1nn-v2",
                                                                    KnnConfigs::buildTunedTwed1nnV2,
                                                                    "similarity", "distance", "univariate"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("cee-v1",
                                                                    CEE::buildV1,
                                                                    "similarity", "distance", "univariate", "ensemble"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("cee-v2",
                                                                    CEE::buildV2,
                                                                    "similarity", "distance", "univariate", "ensemble"));
        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("lee",
                                                                    CEE::buildLee,
                                                                    "similarity", "distance", "univariate", "ensemble"));
    }

    public static void main(String[] args) {
        System.out.println(getInstance().toString());
    }
}
