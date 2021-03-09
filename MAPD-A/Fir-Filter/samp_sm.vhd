LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY samp_sm IS
    PORT (
        clk, uart_rx, pulse_out : IN STD_LOGIC;
        enable : OUT STD_LOGIC
    );
END samp_sm;
ARCHITECTURE rtl OF samp_sm IS
    TYPE state_type IS (idle, start_s, cnt0, cnt1, cnt2, cnt3, cnt4, cnt5, cnt6, cnt7, stp);
    SIGNAL state : state_type := idle;
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF (rising_edge(clk)) THEN
            CASE state IS
                WHEN idle =>
                    enable <= '0';
                    IF uart_rx = '0' THEN
                        state <= start_s;
                    END IF;
                WHEN start_s =>
                    enable <= '1';
                    IF pulse_out = '1' THEN
                        state <= cnt0;
                    END IF;
                WHEN cnt0 =>
                    IF pulse_out = '1' THEN
                        state <= cnt1;
                    END IF;
                WHEN cnt1 =>
                    IF pulse_out = '1' THEN
                        state <= cnt2;
                    END IF;
                WHEN cnt2 =>
                    IF pulse_out = '1' THEN
                        state <= cnt3;
                    END IF;
                WHEN cnt3 =>
                    IF pulse_out = '1' THEN
                        state <= cnt4;
                    END IF;
                WHEN cnt4 =>
                    IF pulse_out = '1' THEN
                        state <= cnt5;
                    END IF;
                WHEN cnt5 =>
                    IF pulse_out = '1' THEN
                        state <= cnt6;
                    END IF;
                WHEN cnt6 =>
                    IF pulse_out = '1' THEN
                        state <= cnt7;
                    END IF;
                WHEN cnt7 =>
                    IF pulse_out = '1' THEN
                        state <= stp;
                    END IF;
                WHEN stp =>
                    enable <= '0';
                    IF uart_rx = '1' THEN
                        state <= idle;
                    END IF;
            END CASE;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;