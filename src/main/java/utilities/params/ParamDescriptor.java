package utilities.params;

import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

public class ParamDescriptor<A> {
    private final String name;
    private final Supplier<A> accessor;
    private final Consumer<A> mutator;
    private final Function<String, A> fromString;
    private final Function<A, String> toString;

    public ParamDescriptor(final String name, final Supplier<A> accessor, final Consumer<A> mutator,
                           final Function<String, A> fromString,
                           final Function<A, String> toString) {
        this.name = name;
        this.accessor = accessor;
        this.mutator = mutator;
        this.fromString = fromString;
        this.toString = toString;
    }

    public ParamDescriptor(final String name, final Supplier<A> accessor, final Consumer<A> mutator,
                           final Function<String, A> fromString) {
        this(name, accessor, mutator, fromString, String::valueOf);
    }

    public String getName() {
        return name;
    }

    public Supplier<A> getAccessor() {
        return accessor;
    }

    public Consumer<A> getMutator() {
        return mutator;
    }

    public Function<String, A> getFromString() {
        return fromString;
    }

    public Function<A, String> getToString() {
        return toString;
    }

    @Override public boolean equals(final Object o) {
        if(this == o) return true;
        if(o == null || !(getClass().equals(o.getClass()))) return false;
        final ParamDescriptor<?> that = (ParamDescriptor<?>) o;
        return Objects.equals(name, that.name);
    }

    @Override public int hashCode() {
        return Objects.hash(name);
    }
}
