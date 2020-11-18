package tsml.data_containers;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TSCapabilities {

    /** the object that owns this capabilities instance */
    protected TSCapabilitiesHandler owner;
    
    /** the set for storing the active capabilities */
    protected Set<TSCapability> capabilities;


    public int numCapabilities(){
        return capabilities.size();
    }

    public TSCapabilities(){
        capabilities = new HashSet<>();
    }

    public TSCapabilities(final TSCapabilitiesHandler owner) {
        this();
        this.owner = owner;        
    }

    public TSCapabilities enable(TSCapability capability){
        capabilities.add(capability);
        return this;
    }

    public TSCapabilities enableOr(TSCapability either, TSCapability or){
        capabilities.add(new Or(either, or));
        return this;
    }

    public TSCapabilities enableAnd(TSCapability either, TSCapability and){
        capabilities.add(new And(either, and));
        return this;
    }

    public TSCapabilities disable(TSCapability capability){
        capabilities.remove(capability);
        return this;
    }
    
    public boolean handles(TSCapability capability){
        return capabilities.contains(capability);
    }

    public boolean test(TimeSeriesInstances data) {
        return this.capabilities.stream().allMatch(e -> e.test(data));
    }

    public boolean test(TimeSeriesInstance inst){
        return this.capabilities.stream().allMatch(e -> e.test(inst));
    }

    public static abstract class TSCapability{
        public abstract boolean test(TimeSeriesInstances data);
        public abstract boolean test(TimeSeriesInstance inst);
    }

    public static TSCapability EQUAL_LENGTH = new EqualLength();
    public static TSCapability UNEQUAL_LENGTH = new Not(new EqualLength());
    public static TSCapability UNIVARIATE = new Not(new Multivariate());
    public static TSCapability MULTIVARIATE = new Multivariate();
    public static TSCapability NO_MISSING_VALUES = new Not(new MissingValues());
    public static TSCapability MISSING_VALUES = new MissingValues();

    public static TSCapability MULTI_OR_UNIVARIATE = new Or(UNIVARIATE, MULTIVARIATE);
    public static TSCapability EQUAL_OR_UNEQUAL_LENGTH = new Or(EQUAL_LENGTH, UNEQUAL_LENGTH);
    
    public static TSCapability MIN_LENGTH(int length){
        return new MinLength(length);
    }

    //adapter wrapper to flip a boolean to simplify capabilities objects.
    private static final class Not extends TSCapability{
        TSCapability cap;
        private Not(TSCapability capability){
            cap = capability;
        }
        @Override
        public boolean test(TimeSeriesInstances data) {
            return !cap.test(data);
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return !cap.test(inst);
        }

        @Override
        public int hashCode(){
            return cap.hashCode();
        }
    }

    private static final class Or extends TSCapability{
        Set<TSCapability> caps;
        private Or(TSCapability... capabilites){
            caps = Stream.of(capabilites).collect(Collectors.toSet());
        }

        @Override
        public boolean test(TimeSeriesInstances data) {
            boolean out = false;
            for(TSCapability cap : caps)
                out |= cap.test(data);

            return out;
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            boolean out = false;
            for(TSCapability cap : caps)
                out |= cap.test(inst);
             
            return out;
        }

        @Override
        public int hashCode(){
            return caps.hashCode();
        }
    }

    private static final class And extends TSCapability{
        Set<TSCapability> caps;
        private And(TSCapability... capabilites){
            caps = Stream.of(capabilites).collect(Collectors.toSet());
        }

        @Override
        public boolean test(TimeSeriesInstances data) {
            boolean out = false;
            for(TSCapability cap : caps)
                out &= cap.test(data);

            return out;
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            boolean out = false;
            for(TSCapability cap : caps)
                out &= cap.test(inst);
             
            return out;
        }

        @Override
        public int hashCode(){
            return caps.hashCode();
        }
    }

    protected static final class EqualLength extends TSCapability{
        @Override
        public boolean test(TimeSeriesInstances data) {
            return data.isEqualLength();
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return inst.isEqualLength();
        }
    }
    
    protected static final class Multivariate extends TSCapability{
        @Override
        public boolean test(TimeSeriesInstances data) {
            return data.isMultivariate();
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return inst.isMultivariate();
        }
    }

    protected static final class MissingValues extends TSCapability{
        @Override
        public boolean test(TimeSeriesInstances data) {
            return data.hasMissing();
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return inst.hasMissing();
        }
    }

    protected static final class MinLength extends TSCapability{

        int minL;

        protected MinLength(int minL){
            this.minL = minL;
        }

        @Override
        public boolean test(TimeSeriesInstances data) {
            return data.getMinLength() >= minL;
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return inst.getMinLength() >= minL;
        }
    }
    
}
