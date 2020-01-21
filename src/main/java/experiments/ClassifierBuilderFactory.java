package experiments;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.knn.configs.KnnConfig;
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

        public ClassifierBuilder(final String name, final Supplier<A> supplier,
                                 final List<Supplier<String>> tagSuppliers) { // todo i'd like this to be an
            // Iterable<Supplier<String>> rather than list but somehow it conflicts with the other constructor!! Odd.
            this.name = name;
            this.supplier = supplier;
            List<String> tags = new ArrayList<>();
            for(Supplier<String> str : tagSuppliers) {
                tags.add(str.get().toLowerCase());
            }
            this.tags = ImmutableList.copyOf(tags);
        }

        public ClassifierBuilder(final String name,
                                 final Supplier<A> supplier,
                                 final Supplier<String>... tagSuppliers) {
            this(name, supplier, Arrays.asList(tagSuppliers));
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

    public static ClassifierBuilderFactory getGlobalInstance() {
        return INSTANCE;
    }

    public void add(Supplier<ClassifierBuilder<?>> supplier) {
        add(supplier.get());
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

    public ClassifierBuilder<?> getClassifierBuilderByName(String name) {
        name = name.toLowerCase();
        ClassifierBuilder<?> classifierBuilder = classifierBuildersByName.get(name);
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

    static {
        getGlobalInstance().add(KnnConfig.ED_1NN_V1);
        getGlobalInstance().add(KnnConfig.ED_1NN_V2);
        getGlobalInstance().add(KnnConfig.DTW_1NN_V1);
        getGlobalInstance().add(KnnConfig.DTW_1NN_V2);
        getGlobalInstance().add(KnnConfig.DDTW_1NN_V1);
        getGlobalInstance().add(KnnConfig.DDTW_1NN_V2);
        getGlobalInstance().add(KnnConfig.TUNED_DTW_1NN_V1);
        getGlobalInstance().add(KnnConfig.TUNED_DTW_1NN_V2);
        getGlobalInstance().add(KnnConfig.TUNED_WDTW_1NN_V1);
        getGlobalInstance().add(KnnConfig.TUNED_WDTW_1NN_V2);
        getGlobalInstance().add(KnnConfig.TUNED_WDDTW_1NN_V1);
        getGlobalInstance().add(KnnConfig.TUNED_WDDTW_1NN_V2);
        getGlobalInstance().add(KnnConfig.TUNED_LCSS_1NN_V1);
        getGlobalInstance().add(KnnConfig.TUNED_LCSS_1NN_V2);
        getGlobalInstance().add(KnnConfig.TUNED_MSM_1NN_V1);
        getGlobalInstance().add(KnnConfig.TUNED_MSM_1NN_V2);
        getGlobalInstance().add(KnnConfig.TUNED_ERP_1NN_V1);
        getGlobalInstance().add(KnnConfig.TUNED_ERP_1NN_V2);
        getGlobalInstance().add(KnnConfig.TUNED_TWED_1NN_V1);
        getGlobalInstance().add(KnnConfig.TUNED_TWED_1NN_V2);
        getGlobalInstance().add(KnnConfig.CEE_V1);
        getGlobalInstance().add(KnnConfig.CEE_V2);
        getGlobalInstance().add(KnnConfig.LEE);
    }

    public static void main(String[] args) {
        System.out.println(getGlobalInstance().toString());
    }
}
