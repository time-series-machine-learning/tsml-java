package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.strings.StrUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class Interval implements ParamHandler {
    private int start = -1;
    private int end = -1;
    public static final String START_FLAG = "s";
    public static final String END_FLAG = "e";

    public Interval(final int start, final int end) {
        setEnd(end);
        setStart(start);
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(final int end) {
        this.end = end;
    }

    public int getStart() {
        return start;
    }

    public void setStart(final int start) {
        this.start = start;
    }

    public int size() {
        return Math.abs(end - start) + 1;
    }

    @Override public ParamSet getParams() {
        return ParamHandler.super.getParams().add(START_FLAG, start).add(END_FLAG, end);
    }

    @Override public void setParams(final ParamSet paramSet) {
        final Object startObj = paramSet.getSingleOrDefault(START_FLAG, start);
//        StrUtils.fromOptionValue(startObj, Integer.class);
        ParamHandler.setParam(paramSet, START_FLAG, this::setStart, Integer.class);
        ParamHandler.setParam(paramSet, END_FLAG, this::setEnd, Integer.class);
    }

    public static <A> A abc(Object value, Function<String, ? extends A> fromString) {
        A result = null;
        if(value instanceof String) {
            fromString.apply((String) value);
        } else {
            result = (A) value;
        }
        return result;
    }

    public static void main(String[] args) {
        ParamSet paramSet = new ParamSet();
        paramSet.add("a",  "5");
        final Object o = paramSet.getSingle("a");

        //        final Integer a1 = paramSet.getSingleOrDefault("a", 6, Integer::parseInt);
//        final Object ao = paramSet.getSingleOrDefault("a", "6");
//        final Integer a = abc(ao, Integer::parseInt);
//        paramSet.add("a", "4");
//        final List<Integer> b = paramSet.getOrDefault("a", new ArrayList<>(Collections.singletonList(6)));
//        System.out.println();
    }
}
