package utilities;

import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class SysUtils {
    public static List<String> exec(String command) throws
                                             IOException,
                                             InterruptedException {
        Process process = Runtime.getRuntime().exec(command);
        List<String> result;
        process.waitFor();
        if(process.exitValue() == 0) {
            result = IOUtils.readLines(process.getInputStream(), StandardCharsets.UTF_8);
        } else {
            throw new IllegalStateException(StringUtilities.join("\n", IOUtils.readLines(process.getErrorStream(),
                                                                                         StandardCharsets.UTF_8)));
        }
        process.destroyForcibly();
        return result;
    }

    public static String findCpuInfo() {
        try {
            OS os = getOS();
            String cpuInfo;
            switch(os) {
                case MAC:
                case LINUX:
                case SOLARIS:
                    cpuInfo = exec("cat /proc/cpuinfo | grep 'model name'").get(0);
                    break;
                case WINDOWS:
                default:
                    cpuInfo = "unknown";
                    break;
            }
            return cpuInfo;
        } catch(IOException | InterruptedException e) {
            return "unknown";
        }
    }

    public static String getOsName() {
        return getOS().name().toLowerCase();
    }

    public enum OS {
        WINDOWS, LINUX, MAC, SOLARIS
    }

    private static OS os = getOS();

    public static OS getOS() {
        if (os == null) {
            String operSys = System.getProperty("os.name").toLowerCase();
            if (operSys.contains("win")) {
                os = OS.WINDOWS;
            } else if (operSys.contains("nix") || operSys.contains("nux")
                || operSys.contains("aix")) {
                os = OS.LINUX;
            } else if (operSys.contains("mac")) {
                os = OS.MAC;
            } else if (operSys.contains("sunos")) {
                os = OS.SOLARIS;
            }
        }
        return os;
    }
}
