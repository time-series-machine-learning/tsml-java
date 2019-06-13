package utilities;

public class Checks {
    public static void percentageCheck(double percentage) {
        if(percentage < 0 || percentage > 1) {
            throw new IllegalArgumentException("percentage out of range: " + percentage);
        }
    }
}
