LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY dff IS
    PORT (
        clk : IN STD_LOGIC;
        rst : IN STD_LOGIC;
        d
        : IN STD_LOGIC;
        q
        : OUT STD_LOGIC);
END ENTITY dff;
ARCHITECTURE rtl OF dff IS
BEGIN -- architecture rtl
    flipflop : PROCESS (clk) IS
    BEGIN -- process flipflop
        IF rising_edge(clk) THEN
            IF rst = '0' THEN
                q <= '0';
            ELSE
                q <= d;
            END IF;
        END IF;
    END PROCESS flipflop;
END ARCHITECTURE rtl;