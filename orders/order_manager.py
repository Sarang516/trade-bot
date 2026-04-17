"""orders/order_manager.py — Built in Phase 7."""
class OrderManager:
    def __init__(self, broker, risk_manager, trade_logger, notifier, settings):
        self.broker = broker
        self.risk = risk_manager
        self.logger = trade_logger
        self.notifier = notifier
        self._settings = settings
    def process_signal(self, signal): pass
    def sync_with_broker(self): pass
    def square_off_all(self): pass
