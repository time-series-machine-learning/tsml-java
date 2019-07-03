package classifiers.template_classifier;

import weka.core.OptionHandler;

import java.util.*;

public class OptionSet implements OptionHandler {

    private final Set<Option> options = new HashSet<>();

    public class Option<A> {
        public A get() {
            return value;
        }

        public void set(A value) {
            this.value = value;
        }

        private A value;

        public String name() {
            return name;
        }

        private final String name;

        public Option(A value, String name) {
            this.value = value;
            this.name = name;
            add(this);
        }

    }

    private void add(Option option) {
        if(options.add(option)) throw new IllegalStateException("cannot have duplicate options");
    }

    @Override
    public String toString() {
        throw new UnsupportedOperationException();
    }


    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}