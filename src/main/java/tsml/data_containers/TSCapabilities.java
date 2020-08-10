package tsml.data_containers;

import java.util.HashSet;
import java.util.Set;

public class TSCapabilities {

    /** the object that owns this capabilities instance */
    protected TSCapabilitiesHandler owner;
    
    /** the set for storing the active capabilities */
    protected Set<TSCapability> capabilities;


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
    }

    protected static final class EqualLength extends TSCapability{
        @Override
        public boolean test(TimeSeriesInstances data) {
            return data.isEqualLength;
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return inst.isEqualLength;
        }
    }

    protected static final class Multivariate extends TSCapability{
        @Override
        public boolean test(TimeSeriesInstances data) {
            return data.isMultivariate;
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return inst.isMultivariate;
        }
    }

    protected static final class MissingValues extends TSCapability{
        @Override
        public boolean test(TimeSeriesInstances data) {
            return data.hasMissing;
        }

        @Override
        public boolean test(TimeSeriesInstance inst) {
            return inst.hasMissing;
        }
    }
    
}