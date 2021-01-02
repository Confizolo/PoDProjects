LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY pulse_gen IS
    PORT (
        clk,enable : IN STD_LOGIC;
        samp_out : OUT STD_LOGIC
    );
END pulse_gen;
ARCHITECTURE rtl OF pulse_gen IS
signal counter : unsigned(9 downto 0):= to_unsigned(867,10);
constant divisor: unsigned(9 downto 0):= to_unsigned(867,10);
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF rising_edge(clk) THEN
            IF enable = '1' THEN 
                counter <= counter + 1;
                IF counter = divisor THEN
                    samp_out <= '1';
                    counter <= (others => '0');
                ELSE
                    samp_out <= '0';
                END IF;
            END IF;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;
