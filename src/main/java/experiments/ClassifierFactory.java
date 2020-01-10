package experiments;

import com.google.common.collect.ImmutableList;
import weka.classifiers.Classifier;

import java.util.*;
import java.util.function.Supplier;

public class ClassifierFactory {

    public static class ClassifierBuilder<A extends Classifier> {
        private final String name;
        private final ImmutableList<String> tags;
        private final Supplier<A> supplier;

        public ClassifierBuilder(final String name,
                                 final ImmutableList<String> tags, final Supplier<A> supplier) {
            this.name = name;
            this.tags = tags;
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
    }

    private static ClassifierFactory INSTANCE = new ClassifierFactory();
    private final Map<String, ClassifierBuilder<?>> classifiersByName = new HashMap<>();
    private final Map<String, Set<ClassifierBuilder<?>>> classifierByTag = new HashMap<>();

    public static ClassifierFactory getInstance() {
        return INSTANCE;
    }

    public void add(ClassifierBuilder<?> classifierBuilder) {
        String name = classifierBuilder.getName();
        ClassifierBuilder<?> current = classifiersByName.get(name);
        if(current != null) {
            throw new IllegalArgumentException("oops, a classifier already exists under the name: " + name);
        } else {
            classifiersByName.put(name, classifierBuilder);
        }
        for(String tag : classifierBuilder.getTags()) {
            classifierByTag.computeIfAbsent(tag, k -> new HashSet<>()).add(classifierBuilder);
        }
    }

    public ClassifierBuilder<?> getClassifierByName(String name) {
        ClassifierBuilder<?> classifierBuilder = classifiersByName.get(name);
        if(classifierBuilder == null) {
            throw new IllegalArgumentException("oops, there's no classifier by the name: " + name);
        }
        return classifierBuilder;
    }

    public Set<ClassifierBuilder<?>> getClassifiersByTags(List<String> tags) {

    }

    public Set<ClassifierBuilder<?>> getClassifiersByTags(String... tags) {
        return getClassifiersByTags(Arrays.asList(tags));
    }
}
