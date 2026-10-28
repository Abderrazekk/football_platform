from django.test import TestCase


class SeasonViewsTests(TestCase):
    def test_best_player_public(self):
        r = self.client.get("/season/best-player/")
        self.assertEqual(r.status_code, 200)

    def test_best_young_public(self):
        r = self.client.get("/season/best-young/")
        self.assertEqual(r.status_code, 200)

    def test_best_xi_public(self):
        r = self.client.get("/season/best-xi/")
        self.assertEqual(r.status_code, 200)

    def test_all_three_render_gracefully_without_artifacts(self):
        # Whatever the state of ml/artifacts/, views must not 500
        for url in ("/season/best-player/", "/season/best-young/", "/season/best-xi/"):
            r = self.client.get(url)
            self.assertIn(r.status_code, (200, 302), f"{url} returned {r.status_code}")