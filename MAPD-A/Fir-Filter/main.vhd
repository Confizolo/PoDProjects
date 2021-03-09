LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY main_sm IS
    PORT (
        clk, uart_rx, baud_in : IN STD_LOGIC;
        valid : OUT STD_LOGIC;
        rec_out : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END main_sm;
ARCHITECTURE rtl OF main_sm IS
    TYPE state_type IS (idle, start_s, rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7);
    SIGNAL state : state_type := idle;
    SIGNAL rec_out_s : STD_LOGIC_VECTOR(7 DOWNTO 0) := (OTHERS => '0');
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF (rising_edge(clk)) THEN
            CASE state IS
                WHEN idle =>
                    rec_out <= (OTHERS => '0');
                    valid <= '0';
                    IF uart_rx = '0' THEN
                        state <= start_s;
                    END IF;
                WHEN start_s =>
                    IF baud_in = '1' THEN
                        rec_out_s(0) <= uart_rx;
                        state <= rd0;
                    END IF;
                WHEN rd0 =>
                    IF baud_in = '1' THEN
                        rec_out_s(1) <= uart_rx;
                        state <= rd1;
                    END IF;
                WHEN rd1 =>
                    IF baud_in = '1' THEN
                        rec_out_s(2) <= uart_rx;
                        state <= rd2;
                    END IF;
                WHEN rd2 =>
                    IF baud_in = '1' THEN
                        rec_out_s(3) <= uart_rx;
                        state <= rd3;
                    END IF;
                WHEN rd3 =>
                    IF baud_in = '1' THEN
                        rec_out_s(4) <= uart_rx;
                        state <= rd4;
                    END IF;
                WHEN rd4 =>
                    IF baud_in = '1' THEN
                        rec_out_s(5) <= uart_rx;
                        state <= rd5;
                    END IF;
                WHEN rd5 =>
                    IF baud_in = '1' THEN
                        rec_out_s(6) <= uart_rx;
                        state <= rd6;
                    END IF;
                WHEN rd6 =>
                    IF baud_in = '1' THEN
                        rec_out_s(7) <= uart_rx;
                        state <= rd7;
                    END IF;
                WHEN rd7 =>
                    IF baud_in = '1' THEN
                        rec_out <= rec_out_s;
                        valid <= '1';
                        state <= idle;
                    END IF;
            END CASE;

        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;