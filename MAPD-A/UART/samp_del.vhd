LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY samp_del IS
    PORT (
        clk, pulse_out : IN STD_LOGIC;
        baud_out : OUT STD_LOGIC
    );
END samp_del;
ARCHITECTURE rtl OF samp_del IS
    SIGNAL counter : unsigned(9 DOWNTO 0) := (OTHERS => '0');
    CONSTANT divisor : unsigned(9 DOWNTO 0) := to_unsigned(433, 10);
    SIGNAL buff : STD_LOGIC;
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF (rising_edge(clk)) THEN
            IF pulse_out = '1' THEN
                buff <= '1';
            END IF;
            IF buff = '1' THEN
                counter <= counter + 1;
                IF counter = divisor THEN
                    baud_out <= '1';
                    buff <= '0';
                    counter <= (OTHERS => '0');
                ELSE
                    baud_out <= '0';
                END IF;
            ELSE
                baud_out <= '0';
            END IF;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;