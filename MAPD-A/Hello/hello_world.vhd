USE std.textio.ALL;
ENTITY hello_world IS
END hello_world;
ARCHITECTURE behaviour OF hello_world IS BEGIN
    PROCESS
        VARIABLE l : line;
    BEGIN
        write(l, STRING'("Hello world!"));
        writeline (output, l);
        WAIT;
    END PROCESS;
END behaviour;