package classifiers.template_classifier;

import weka.core.OptionHandler;

import java.util.*;
import java.util.function.Function;

public class OptionSet implements OptionHandler {

    private final Map<String, Option> options = new HashMap<>();

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
        private final Function<String, A> fromString;

        public void set(String string) {
            set(fromString.apply(string)); // todo this might be problematic with a string type of A
        }

        public Option(A value, String name, final Function<String, A> fromString) {
            this.value = value;
            this.name = name;
            this.fromString = fromString;
            addUnique(this);
        }

    }

    private void add(Option option) {
        options.put(option.name(), option);
    }

    private void addUnique(Option option) {
        Option current = options.get(option.name());
        if(current == null) {
            add(option);
        } else {
            throw new IllegalStateException("cannot have duplicate options");
        }
    }

    @Override
    public String toString() {
        throw new UnsupportedOperationException();
    }


    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    public void setOption(String key, String value) {
        Option option = options.get(key);
        if(option == null) {
            throw new IllegalArgumentException("unknown option: " + key);
        } else {
            option.set(value);
        }
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        if(options.length % 2 != 0) {
            throw new IllegalArgumentException("options must be an array of key value pairs, i.e. even in length");
        }
        for(int i = 0; i < options.length; i+=2) {
            setOption(options[i], options[i + 1]);
        }
    }

    @Override
    public String[] getOptions() {
        String[] optionsString = new String[options.size() * 2];
        int i = 0;
        for(Map.Entry<String, Option> entry : options.entrySet()) {
            optionsString[i++] = entry.getKey();
            optionsString[i++] = String.valueOf(entry.getValue().get());
        }
        return optionsString;
    }

    public String getOption(String key) {
        Option option = options.get(key);
        if(option == null) {
            throw new IllegalArgumentException("unknown option: " + key);
        } else {
            return String.valueOf(option.get());
        }
    }
}
