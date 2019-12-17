package utilities.params;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.Dtw;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Param {
    public static class ParamValue {
        private Object value;
        private List<Param> paramList = new ArrayList<>();

        public ParamValue() {}

        public ParamValue(Object value, List<Param> paramList) {
            setParamList(paramList);
            setValue(value);
        }

        public ParamValue(Object value) {
            this(value, null); // no sub param
        }

        public Object getValue() {
            return value;
        }

        public void setValue(final Object value) {
            this.value = value;
        }

        public List<Param> getParamList() {
            return paramList;
        }

        public void setParamList(List<Param> paramList) {
            if(paramList == null) {
                paramList = new ArrayList<>();
            }
            this.paramList = paramList;
        }

        public void addParam(final Param param) {
            paramList.add(param);
        }

        @Override public String toString() {
            return "ParamValue{" +
                "value=" + value +
                ", paramList=" + paramList +
                '}';
        }
    }

    private Map<String, List<ParamValue>> paramMap = new HashMap<>();

    public void add(String name, ParamValue value) {
        paramMap.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
    }

    @Override public String toString() {
        return "Param{" +
            "paramMap=" + paramMap +
            '}';
    }

    public static void main(String[] args) {
//        Param param = new Param();
//        Param wParam = new Param();
//        wParam.add(Dtw.WARPING_WINDOW_FLAG, new ParamValue(4));
//        param.add(DistanceMeasure.DISTANCE_FUNCTION_FLAG, new ParamValue(new Dtw(), wParam));

    }
}
