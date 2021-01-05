LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY pulse_gen IS
    PORT (
        clk, enable : IN STD_LOGIC;
        samp_out : OUT STD_LOGIC
    );
END pulse_gen;
ARCHITECTURE rtl OF pulse_gen IS
    SIGNAL out_val : STD_LOGIC;
    SIGNAL counter : unsigned(9 DOWNTO 0) := (OTHERS => '0');
    CONSTANT divisor : unsigned(9 DOWNTO 0) := to_unsigned(867, 10);
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF rising_edge(clk) THEN
            IF enable = '1' THEN
                samp_out <= out_val;
                counter <= counter + 1;
                IF counter = divisor THEN
                    out_val <= '1';
                    counter <= (OTHERS => '0');
                ELSE
                    out_val <= '0';
                END IF;
            ELSE
                out_val <= '1';
            END IF;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;